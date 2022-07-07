# -*- coding:utf-8 _*-
import os
import logging
import torch
import inspect
import random
import numpy as np
from tqdm import tqdm
from config import parse
from typing import Optional
from matches import matches
from functools import partial
from datasets import load_dataset
from utils import divide_parameters
from utils_glue import compute_metrics
from transformers import BertConfig, BertTokenizerFast, default_data_collator
from transformers import BertForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from textbrewer import DistillationConfig, TrainingConfig, GeneralDistiller
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")


def remove_unused_columns(model, dataset, description: Optional[str] = None):
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += ["label", "label_ids"]
    columns = [k for k in signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    dset_description = "" if description is None else f"in the {description} set "
    logger.info(
        f"The following columns {dset_description}don't have a corresponding argument in `{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
    )
    dataset.set_format(type=dataset.format["type"], columns=columns)


def args_check(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu


def BertForGLUESimpleAdaptor(batch, model_outputs, no_logits, no_mask):
    dict_obj = {'hidden': model_outputs[2], 'attention': model_outputs[3]}
    if no_mask is False:
        dict_obj['attention_mask'] = batch['attention_mask']
    if no_logits is False:
        dict_obj['logits'] = (model_outputs[1],)
    return dict_obj


def predict(model, eval_dataset, step, args):
    eval_output_dir = args.output_dir
    results = {}
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    logger.info("Predicting...")
    logger.info("***** Running predictions *****")
    logger.info("  Num  examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.predict_batch_size)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    remove_unused_columns(model, eval_dataset, description="evaluation")
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.predict_batch_size,
        collate_fn=default_data_collator,
    )
    model.eval()

    pred_logits = []
    label_ids = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
        labels = batch['labels']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        cpu_logits = logits[0].detach().cpu()
        for i in range(len(cpu_logits)):
            pred_logits.append(cpu_logits[i].numpy())
            label_ids.append(labels[i])

    pred_logits = np.array(pred_logits)
    label_ids = np.array(label_ids)

    if args.output_mode == "classification":
        preds = np.argmax(pred_logits, axis=1)
    else:  # args.output_mode == "regression":
        preds = np.squeeze(pred_logits)
    result = compute_metrics('wnli', preds, label_ids)
    logger.info(f"result: {result}")
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} task {} *****".format(step, ''))
        writer.write("step: %d ****\n " % step)
        for key in sorted(results.keys()):
            logger.info("%s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))
    model.train()
    return results


def main():
    args = parse()
    for k, v in vars(args).items():
        logger.info(f"{k}:{v}")
    # set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # arguments check
    device, n_gpu = args_check(args)
    os.makedirs(args.output_dir, exist_ok=True)
    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size
    # load bert config
    bert_config_t = BertConfig.from_json_file(args.bert_config_file_T)
    bert_config_s = BertConfig.from_json_file(args.bert_config_file_S)
    assert args.max_seq_length <= bert_config_t.max_position_embeddings
    assert args.max_seq_length <= bert_config_s.max_position_embeddings

    # Prepare GLUE task
    args.output_mode = "classification"
    num_labels = args.num_labels
    # read data
    num_train_steps = None
    datasets = load_dataset(
        "csv", data_files={"train": args.data_dir + '/train.csv',
                           "validation": args.data_dir + '/dev.csv',
                           "test": args.data_dir + '/test.csv'}, delimiter='\t')
    # Preprocessing the datasets
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 8:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[1], None

    padding = "max_length"
    max_length = args.max_seq_length

    tokenizer = BertTokenizerFast(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    train_dataset = datasets["train"]
    eval_datasets = datasets["validation"]


    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.do_train:
        num_train_steps = int(len(train_dataset) / args.train_batch_size) * args.num_train_epochs
    logger.info("Data loaded")

    # Build Model and load checkpoint
    bert_config_t.num_labels = num_labels
    bert_config_s.num_labels = num_labels
    bert_config_t.output_attentions = args.output_attention_layers
    bert_config_t.output_hidden_states = args.output_encoded_layers
    bert_config_s.output_attentions = args.output_attention_layers
    bert_config_s.output_hidden_states = args.output_encoded_layers
    model_T = BertForSequenceClassification(bert_config_t)  # bert-base
    model_S = BertForSequenceClassification(bert_config_s)  # hfl-rbt3

    # Load teacher
    if args.tuned_checkpoint_T is not None:
        state_dict_T = torch.load(args.tuned_checkpoint_T, map_location='cpu')
        model_T.load_state_dict(state_dict_T)
        model_T.eval()
    else:
        assert args.do_predict is True

    # Load student
    if args.load_model_type == 'bert':
        assert args.init_checkpoint_S is not None
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        if args.only_load_embedding:
            state_weight = {k[5:]: v for k, v in state_dict_S.items() if k.startswith('bert.embeddings')}
            missing_keys, _ = model_S.bert.load_state_dict(state_weight, strict=False)
            logger.info(f"Missing keys {list(missing_keys)}")
        else:
            state_weight = {k[5:]: v for k, v in state_dict_S.items() if k.startswith('bert.')}
            missing_keys, _ = model_S.bert.load_state_dict(state_weight, strict=False)
        logger.info("Model loaded")
    elif args.load_model_type == 'all':
        assert args.tuned_checkpoint_S is not None
        state_dict_S = torch.load(args.tuned_checkpoint_S, map_location='cpu')
        model_S.load_state_dict(state_dict_S)
        logger.info("Model loaded")
    else:
        logger.info("Model is randomly initialized.")
    model_T.to(device)
    model_S.to(device)

    # 多卡分布式训练
    if args.local_rank != -1 or n_gpu > 1:
        if args.local_rank != -1:
            raise NotImplementedError
        elif n_gpu > 1:
            model_T = torch.nn.DataParallel(model_T)  # ,output_device=n_gpu-1)
            model_S = torch.nn.DataParallel(model_S)  # ,output_device=n_gpu-1)

    if args.do_train:
        # parameters  学生参数
        params = list(model_S.named_parameters())
        all_trainable_params = divide_parameters(args, params, lr=args.learning_rate)
        logger.info("Length of all_trainable_params: %d", len(all_trainable_params))
        optimizer = AdamW(all_trainable_params, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(num_train_steps * args.warmup_proportion),
                                                    num_training_steps=num_train_steps)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Forward batch size = %d", forward_batch_size)
        logger.info("  Num backward steps = %d", num_train_steps)

        """DISTILLATION"""
        train_config = TrainingConfig(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ckpt_frequency=args.ckpt_frequency,
            log_dir=args.output_dir,
            output_dir=args.output_dir,
            device=args.device)

        intermediate_matches = None
        if isinstance(args.matches, (list, tuple)):
            intermediate_matches = []
            for match in args.matches:
                intermediate_matches += matches[match]
        logger.info(f"{intermediate_matches}")
        distill_config = DistillationConfig(
            temperature=args.temperature,
            intermediate_matches=intermediate_matches)
        logger.info(f"{train_config}")
        logger.info(f"{distill_config}")
        adaptor_T = partial(BertForGLUESimpleAdaptor, no_logits=args.no_logits, no_mask=args.no_inputs_mask)
        adaptor_S = partial(BertForGLUESimpleAdaptor, no_logits=args.no_logits, no_mask=args.no_inputs_mask)

        distiller = GeneralDistiller(train_config=train_config,
                                     distill_config=distill_config,
                                     model_T=model_T, model_S=model_S,
                                     adaptor_T=adaptor_T,
                                     adaptor_S=adaptor_S)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            raise NotImplementedError

        remove_unused_columns(model_S, train_dataset, description="training")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.forward_batch_size,
            sampler=train_sampler,
            collate_fn=default_data_collator,
            drop_last=True,
        )
        callback_func = partial(predict, eval_dataset=eval_datasets, args=args)
        with distiller:
            distiller.train(optimizer, scheduler=scheduler, dataloader=train_dataloader,
                            num_epochs=args.num_train_epochs, callback=callback_func)

    if not args.do_train and args.do_predict:
        res = predict(model_S, eval_datasets, step=0, args=args)
        print(res)


if __name__ == '__main__':
    main()
