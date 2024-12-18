import argparse
import functools
import itertools
import os.path
import time

import torch

import numpy as np

from benepar import char_lstm
from benepar import decode_chart
from benepar import nkutil
from benepar import parse_chart
import evaluate
import learning_rates
import treebanks
import nltk


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def make_hparams():
    return nkutil.HParams(
        # Data processing
        max_len_train=0,  # no length limit
        max_len_dev=0,  # no length limit
        # Optimization
        batch_size=32,
        learning_rate=0.00005,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0.0,  # no clipping
        checks_per_epoch=4,
        step_decay_factor=0.5,
        step_decay_patience=500,
        max_consecutive_decays=3,  # establishes a termination criterion
        # CharLSTM
        use_chars_lstm=False,
        d_char_emb=64,
        char_lstm_input_dropout=0.2,
        # BERT and other pre-trained models
        use_pretrained=False,
        pretrained_model="bert-base-uncased",
        # Partitioned transformer encoder
        use_encoder=False,
        d_model=1024,
        num_layers=8,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
        encoder_max_len=512,
        # Dropout
        morpho_emb_dropout=0.2,
        attention_dropout=0.2,
        relu_dropout=0.1,
        residual_dropout=0.2,
        # Output heads and losses
        force_root_constituent="auto",
        predict_tags=False,
        d_label_hidden=256,
        d_tag_hidden=256,
        tag_loss_scale=5.0,
    )


def calculate_f1(pred,gold):

    gold_seq=[str(i) for i in gold]
    pred_seq=[str(i) for i in pred]


    gold_list=[traverse_tree(i) for i in gold]
    pred_list=[traverse_tree(i) for i in pred]



    quad_true_num=0
    quad_pred_num=0
    quad_gold_num=0


    pair_true_num=0
    pair_pred_num=0
    pair_gold_num=0



    aspect_gold_num_set=0
    opinion_gold_num_set=0


    for p,g in zip(pred_list,gold_list):




        quad_gold_num+=len(set(g))
        quad_pred_num+=len(set(p))
        quad_true_num+=len(set(g) & set(p))



    quad_p=quad_true_num/quad_pred_num if quad_true_num!=0 else 0
    quad_r=quad_true_num/quad_gold_num if quad_true_num!=0 else 0
    quad_f1=(2*quad_p*quad_r)/(quad_p+quad_r) if quad_true_num!=0 else 0


    print("Quad:")
    print("Gold Num:{0}  Pred Num:{1}  True Num:{2} :".format(quad_gold_num,quad_pred_num,quad_true_num))
    print("Precision:{0}".format(quad_p))
    print("Recall:{0}".format(quad_r))
    print("F1:{0}".format(quad_f1))

    return quad_f1,gold_list,pred_list,gold_seq,pred_seq

def saveOutput(pred,gold,pred_seq,gold_seq):

    pred_out=pred#[[i.replace("▁"," ") for i in j] for j in pred]
    gold_out=gold#[[i.replace("▁"," ") for i in j] for j in gold]
    #print(pred_out)
    f=open("AMRtoABSAoutput_lap.txt","w")

    for p,g,t,o in zip(pred_out,gold_out,pred_seq,gold_seq):

#             f.write("input:\n")
#             f.write(i)
#             f.write("\n")

        f.write("ground_truth:\n")
        f.write(o)
        f.write("\n")
        f.write("\n")

        f.write("output:\n")
        f.write(t)
        f.write("\n")
        f.write("\n")

        f.write("pred:\n")
        for sp in p:
            f.write(str(sp)+"\n")
        f.write("\n")
        f.write("\n")

        f.write("gold:\n")
        for sg in g:
            f.write(str(sg)+"\n")
        f.write("\n\n\n")
        f.write("\n")
        f.write("\n")

    f.close()

def saveOutputQuad(pred,gold,pred_seq,gold_seq):

    pred_out=pred#[[i.replace("▁"," ") for i in j] for j in pred]
    gold_out=gold#[[i.replace("▁"," ") for i in j] for j in gold]
    #print(pred_out)
    f=open("test_quads.txt","w")

    for p,g,t,o in zip(pred_out,gold_out,pred_seq,gold_seq):

        for sp in p:
            f.write(str(sp)+"###")
        f.write("\n")
    f.close()

def getAO(tree,aspect,opinion,path):
    #print(tree)

    if type(tree) != nltk.tree.Tree:
        return 
    if tree.label()!="Q":
        path.append(tree.label())

    if "A" in [subtree.label() for subtree in tree]:
        a=tree.leaves()
        #path.append(" ".join(a))
        aspect.append([path[:]," ".join(a)])
        if path:
            path.pop(-1)  
        # if path:
        #     path.pop(-1)
        return

    elif "O" in [subtree.label() for subtree in tree]:
        o=tree.leaves()
        #path.append(" ".join(o))
        opinion.append([path[:]," ".join(o)])
        if path:
            path.pop(-1)  
        # if path:
        #     path.pop(-1)
        return

    for subtree in tree:
        if subtree.label()!="W":
            getAO(subtree,aspect,opinion,path)
    if tree.label()!="Q":
        if path:
            path.pop(-1)
def traverse_tree(tree):
    """
    深度遍历nltk.tree
    :param tree: nltk.tree对象
    :return: 无
    """
    #print("quad_list in tree:", tree)
    #print('___________________')
    quad_list=[]
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree and subtree.label()=="Q":

            aspect=[]
            opinion=[]
            path=[]

            getAO(subtree,aspect,opinion,path)

#             print(aspect)

#             print(opinion)
#             print()

            if len(opinion)!=0:
                new_aspect=[]
                for a in aspect:
                    if len(a[0])==1:
                        new_aspect.append([a[0][0],a[1]])
                    else:
                        for c in a[0]:
                            new_aspect.append([c,a[1]])
                aspect=new_aspect
            else:
                aspect=[a[0]+[a[1]] for a in aspect]


            if len(aspect)!=0:
                new_opinion=[]
                for o in opinion:
                    if len(o[0])==1:
                        new_opinion.append([o[0][0],o[1]])
                    else:
                        for p in o[0]:
                            new_opinion.append([p,o[1]])
                opinion=new_opinion
            else:
                opinion=[o[0]+[o[1]] for o in opinion]




            #print(aspect)
            #print(opinion)

            if len(aspect)==0 and len(opinion)==0:
                continue

            elif len(aspect)==0 and len(opinion)!=0:
                for o in opinion:
                    quad_list.append(tuple(["NULL"]+o))

            elif len(aspect)!=0 and len(opinion)==0:
                for a in aspect:
                    quad_list.append(tuple(a+["NULL"]))

            elif len(aspect) < len(opinion):
                for o in opinion:
                    quad_list.append(tuple(aspect[0]+o))

            elif len(aspect) > len(opinion):
                for a in aspect:
                    quad_list.append(tuple(a+opinion[0]))

            elif len(aspect) == len(opinion):
                for a,o in zip(aspect,opinion):
                    quad_list.append(tuple(a+o))

            #traverse_tree(subtree)
    #print(set(quad_list))
    return set(quad_list)
def run_train(args, hparams):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    # Make sure that pytorch is actually being initialized randomly.
    # On my cluster I was getting highly correlated results from multiple
    # runs, but calling reset_parameters() changed that. A brief look at the
    # pytorch source code revealed that pytorch initializes its RNG by
    # calling std::random_device, which according to the C++ spec is allowed
    # to be deterministic.
    seed_from_numpy = np.random.randint(2147483648)
    seed_from_numpy = 1455963724#1566850427#
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)
    print("Hyperparameters:")
    hparams.print()

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = treebanks.load_trees(
        args.train_path, args.train_path_text, args.text_processing
    )
    if hparams.max_len_train > 0:
        train_treebank = train_treebank.filter_by_length(hparams.max_len_train)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = treebanks.load_trees(
        args.dev_path, args.dev_path_text, args.text_processing
    )
    if hparams.max_len_dev > 0:
        dev_treebank = dev_treebank.filter_by_length(hparams.max_len_dev)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Constructing vocabularies...")
    label_vocab = decode_chart.ChartDecoder.build_vocab(train_treebank.trees)
    if hparams.use_chars_lstm:
        char_vocab = char_lstm.RetokenizerForCharLSTM.build_vocab(train_treebank.sents)
    else:
        char_vocab = None

    tag_vocab = set()
    for tree in train_treebank.trees:
        for _, tag in tree.pos():
            tag_vocab.add(tag)
    tag_vocab = ["UNK"] + sorted(tag_vocab)
    tag_vocab = {label: i for i, label in enumerate(tag_vocab)}

    if hparams.force_root_constituent.lower() in ("true", "yes", "1"):
        hparams.force_root_constituent = True
    elif hparams.force_root_constituent.lower() in ("false", "no", "0"):
        hparams.force_root_constituent = False
    elif hparams.force_root_constituent.lower() == "auto":
        hparams.force_root_constituent = (
            decode_chart.ChartDecoder.infer_force_root_constituent(train_treebank.trees)
        )
        print("Set hparams.force_root_constituent to", hparams.force_root_constituent)

    print("Initializing model...")
    print(tag_vocab)
    print(label_vocab)
    print(char_vocab)
    parser = parse_chart.ChartParser(
        tag_vocab=tag_vocab,
        label_vocab=label_vocab,
        char_vocab=char_vocab,
        hparams=hparams,
    )
    
    total = sum([param.nelement() for param in parser.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))
    
    if args.parallelize:
        parser.parallelize()
    elif torch.cuda.is_available():
        parser.cuda()
    else:
        print("Not using CUDA!")

    print("Initializing optimizer...")
    trainable_parameters = [
        param for param in parser.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=hparams.learning_rate, betas=(0.9, 0.98), eps=1e-9
    )

    scheduler = learning_rates.WarmupThenReduceLROnPlateau(
        optimizer,
        hparams.learning_rate_warmup_steps,
        mode="max",
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience * hparams.checks_per_epoch,
        verbose=True,
    )

    clippable_parameters = trainable_parameters
    grad_clip_threshold = (
        np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm
    )

    print("Training...")
    total_processed = 0
    current_processed = 0
    check_every = len(train_treebank) / hparams.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None
    best_dev_processed = 0

    start_time = time.time()

    
    
#     def getAO(tree,aspect,opinion,path):
#     #print(tree)
        
#         if type(tree) != nltk.tree.Tree:
#             return 
#         if tree.label()!="Q":
#             path.append(tree.label())

#         if "A" in [subtree.label() for subtree in tree]:
#             a=tree.leaves()
#             path.append(" ".join(a))
#             aspect.append(path[:])
#             if path:
#                 path.pop(-1)  
#             if path:
#                 path.pop(-1)
#             return

#         elif "O" in [subtree.label() for subtree in tree]:
#             o=tree.leaves()
#             path.append(" ".join(o))
#             opinion.append(path[:])
#             if path:
#                 path.pop(-1)  
#             if path:
#                 path.pop(-1)
#             return

#         for subtree in tree:
#             if subtree.label()!="W":
#                 getAO(subtree,aspect,opinion,path)
#         if tree.label()!="Q":
#             if path:
#                 path.pop(-1)
#     def traverse_tree(tree):
#         """
#         深度遍历nltk.tree
#         :param tree: nltk.tree对象
#         :return: 无
#         """
#         #print("quad_list in tree:", tree)
#         #print('___________________')
#         quad_list=[]
#         for subtree in tree:
#             if type(subtree) == nltk.tree.Tree and subtree.label()=="Q":

#                 aspect=[]
#                 opinion=[]
#                 path=[]

#                 getAO(subtree,aspect,opinion,path)

#                 if len(aspect)==0 and len(opinion)==0:
#                     continue

#                 elif len(aspect)==0 and len(opinion)!=0:
#                     for o in opinion:
#                         quad_list.append(tuple(["NULL"]+o))

#                 elif len(aspect)!=0 and len(opinion)==0:
#                     for a in aspect:
#                         quad_list.append(tuple(a+["NULL"]))

#                 elif len(aspect) < len(opinion):
#                     for o in opinion:
#                         quad_list.append(tuple(aspect[0]+o))

#                 elif len(aspect) > len(opinion):
#                     for a in aspect:
#                         quad_list.append(tuple(a+opinion[0]))

#                 elif len(aspect) == len(opinion):
#                     for a,o in zip(aspect,opinion):
#                         quad_list.append(tuple(a+o))

#                 #traverse_tree(subtree)
#         #print(set(quad_list))
#         return set(quad_list)

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path
        nonlocal best_dev_processed

        dev_start_time = time.time()

        dev_predicted = parser.parse(
            dev_treebank.without_gold_annotations(),
            subbatch_max_tokens=args.subbatch_max_tokens,
        )
        
        quad_f1,gold_list,pred_list,gold_seq,pred_seq=calculate_f1(dev_predicted,dev_treebank.trees)
        
        # print(len(dev_predicted))
        # print(dev_treebank.trees[3])
        #print(traverse_tree(dev_treebank.trees[3]))
        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                quad_f1,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )
        
        if quad_f1 > best_dev_fscore:
            
            
            
            saveOutput(pred_list,gold_list,pred_seq,gold_seq)
            
            
            if best_dev_model_path is not None:
                extensions = ["for_test_speed.pt"]
                for ext in extensions:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = quad_f1
            best_dev_model_path = "{}_bert_dev={:.2f}".format(
                args.model_path_base, quad_f1
            )
            best_dev_processed = total_processed
        
            if quad_f1>=0.39:
                print("Saving new best model to {}...".format(best_dev_model_path))


                torch.save(
                    {
                        "config": parser.config,
                        "state_dict": parser.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    best_dev_model_path + "for_test_speed.pt",
                )
            
            
        print("current best f1 {}".format(best_dev_fscore))
            

    data_loader = torch.utils.data.DataLoader(
        train_treebank,
        batch_size=hparams.batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            parser.encode_and_collate_subbatches,
            subbatch_max_tokens=args.subbatch_max_tokens,
        ),
    )
    for epoch in itertools.count(start=1):
        epoch_start_time = time.time()

        for batch_num, batch in enumerate(data_loader, start=1):
            optimizer.zero_grad()
            parser.train()

            batch_loss_value = 0.0
            for subbatch_size, subbatch in batch:
                loss = parser.compute_loss(subbatch)
                loss_value = float(loss.data.cpu().numpy())
                batch_loss_value += loss_value
                if loss_value > 0:
                    loss.backward()
                del loss
                total_processed += subbatch_size
                current_processed += subbatch_size

            grad_norm = torch.nn.utils.clip_grad_norm_(
                clippable_parameters, grad_clip_threshold
            )

            optimizer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    batch_num,
                    int(np.ceil(len(train_treebank) / hparams.batch_size)),
                    total_processed,
                    batch_loss_value,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()
                scheduler.step(metrics=best_dev_fscore)
            else:
                scheduler.step()

        if (total_processed - best_dev_processed) > (
            (hparams.step_decay_patience + 1)
            * hparams.max_consecutive_decays
            * len(train_treebank)
        ):
            print("Terminating due to lack of improvement in dev fscore.")
            break


def run_test(args):
    
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = treebanks.load_trees(
        args.test_path, args.test_path_text, args.text_processing
    )
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    if len(args.model_path) != 1:
        raise NotImplementedError(
            "Ensembling multiple parsers is not "
            "implemented in this version of the code."
        )

    model_path = args.model_path[0]
    print("Loading model from {}...".format(model_path))
    parser = parse_chart.ChartParser.from_trained(model_path)
    
    total = sum([param.nelement() for param in parser.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))
    
    
    if args.no_predict_tags and parser.f_tag is not None:
        print("Removing part-of-speech tagging head...")
        parser.f_tag = None
    if args.parallelize:
        parser.parallelize()
    elif torch.cuda.is_available():
        
        parser.cuda()

    print("Parsing test sentences...")
    
    print("--------------------------------------------------------")
    start_time = time.time()

    test_predicted = parser.parse(
        test_treebank.without_gold_annotations(),
        subbatch_max_tokens=args.subbatch_max_tokens,
    )
    
    quad_f1,gold_list,pred_list,gold_seq,pred_seq=calculate_f1(test_predicted,test_treebank.trees)
    saveOutputQuad(pred_list,gold_list,pred_seq,gold_seq)

    if args.output_path == "-":
        for tree in test_predicted:
            print(tree.pformat(margin=1e100))
    elif args.output_path:
        with open(args.output_path, "w") as outfile:
            for tree in test_predicted:
                outfile.write("{}\n".format(tree.pformat(margin=1e100)))

    # The tree loader does some preprocessing to the trees (e.g. stripping TOP
    # symbols or SPMRL morphological features). We compare with the input file
    # directly to be extra careful about not corrupting the evaluation. We also
    # allow specifying a separate "raw" file for the gold trees: the inputs to
    # our parser have traces removed and may have predicted tags substituted,
    # and we may wish to compare against the raw gold trees to make sure we
    # haven't made a mistake. As far as we can tell all of these variations give
    # equivalent results.
    ref_gold_path = args.test_path
    if args.test_path_raw is not None:
        print("Comparing with raw trees from", args.test_path_raw)
        ref_gold_path = args.test_path_raw

    print(ref_gold_path)
    test_fscore = evaluate.evalb(
        args.evalb_dir, test_treebank.trees, test_predicted, ref_gold_path=ref_gold_path
    )
    
    print("--------------------------------------------------------")
    print(time.time()-start_time)
    
    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", default="data/absa/lap_train.txt")
    subparser.add_argument("--train-path-text", type=str)
    subparser.add_argument("--dev-path", default="data/absa/lap_test.txt")
    subparser.add_argument("--dev-path-text", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--parallelize", action="store_true")
    subparser.add_argument("--print-vocabs", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path", nargs="+", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="data/absa/lap_test.txt")
    subparser.add_argument("--test-path-text", type=str)
    subparser.add_argument("--test-path-raw", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--subbatch-max-tokens", type=int, default=500)
    subparser.add_argument("--parallelize", action="store_true")
    subparser.add_argument("--output-path", default="")
    subparser.add_argument("--no-predict-tags", action="store_true")

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
