import os, sys
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"  #
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

# torch.backends.cudnn.benchmark = True  # Provides a speedup


import util
import test
import parser_1
import commons
import datasets_ws
from model import network
from model.sync_batchnorm import convert_model
from model.functional import sare_ind, sare_joint
from tensorboardX import SummaryWriter


def run_train():
    #### Initial setup: parser, logging...
    args = parser_1.parse_arguments()
    start_time = datetime.now()
    if not args.test:
        if not os.path.exists(args.runpath):
            os.makedirs(args.runpath)
            print("runspath is not exist,making it")
        writer = SummaryWriter(logdir=join(args.runpath, args.dataset_name + start_time.strftime('%Y-%m-%d_%H-%M-%S')))
    args.save_dir = join("logs", args.save_dir + "_" + start_time.strftime(
        '%Y-%m-%d_%H-%M-%S'))  # , start_time.strftime('%Y-%m-%d_%H-%M-%S')
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    #### Creation of Datasets
    logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

    triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train",
                                              args.negs_num_per_query)
    logging.info(f"Train query set: {triplets_ds}")

    val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
    logging.info(f"Val set: {val_ds}")

    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    logging.info(f"Test set: {test_ds}")

    #### Initialize model
    model = network.GeoLocalizationNetRerank(args)
    model = model.to(args.device)
    if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
        if not args.resume:
            triplets_ds.is_inference = True
            model.aggregation.initialize_netvlad_layer(args, triplets_ds, model.backbone)
        args.features_dim *= args.netvlad_clusters

    Reranker = torch.nn.DataParallel(model.Reranker)
    model = torch.nn.DataParallel(model)
    if args.num_classes == 2:
        CE = torch.nn.CrossEntropyLoss(ignore_index=-100).cuda()
    elif args.num_classes == 1:
        CE = torch.nn.BCEWithLogitsLoss()
    sm = torch.nn.Softmax(dim=1).cuda()
    relu = torch.nn.ReLU().cuda()

    # ============================================================
    for name, param in model.named_parameters():
        if name.startswith('module.backbone.blocks') and int(name[23]) < args.freeze:
            param.requires_grad = False
        if args.fix and (not 'local_head' in name) and (not 'Reranker' in name):
            param.requires_grad = False
        if param.requires_grad:
            print(name)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(model.module.local_head.weight.requires_grad)
    # ======================================================
    #### Setup Optimizer and Loss
    if args.aggregation == "crn":
        crn_params = list(model.module.aggregation.crn.parameters())
        net_params = list(model.module.backbone.parameters()) + \
                     list([m[1] for m in model.module.aggregation.named_parameters() if not m[0].startswith('crn')])
        if args.optim == "adam":
            optimizer = torch.optim.Adam([{'params': crn_params, 'lr': args.lr_crn_layer},
                                          {'params': net_params, 'lr': args.lr_crn_net}])
            logging.info("You're using CRN with Adam, it is advised to use SGD")
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(
                [{'params': crn_params, 'lr': args.lr_crn_layer, 'momentum': 0.9, 'weight_decay': 0.001},
                 {'params': net_params, 'lr': args.lr_crn_net, 'momentum': 0.9, 'weight_decay': 0.001}])
    else:
        if args.optim == "adam":
            optimizer = torch.optim.Adam(parameters, lr=args.lr)  # model.parameters()
        elif args.optim == 'adamw':
            optimizer = torch.optim.AdamW(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                                          amsgrad=False)
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

    if args.criterion == "triplet":
        criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
    elif args.criterion == "sare_ind":
        criterion_triplet = sare_ind
    elif args.criterion == "sare_joint":
        criterion_triplet = sare_joint

    #### Resume model, optimizer, and other training parameters
    if args.resume:
        if args.aggregation != 'crn':
            # model, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
            model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
        else:
            # CRN uses pretrained NetVLAD, then requires loading with strict=False and
            # does not load the optimizer from the checkpoint file.
            model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
        logging.info(f"Resuming from epoch {start_epoch_num} with best recall {best_r5:.1f}")
        best_r5 = start_epoch_num = not_improved_num = 0
    else:
        best_r5 = start_epoch_num = not_improved_num = 0

    # if args.backbone.startswith('vit'):
    #     logging.info(f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}")
    # else:
    #     logging.info(f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}")

    if torch.cuda.device_count() >= 2:
        # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
        model = convert_model(model)
        model = model.cuda()
    if args.test:
        recalls, recalls_str = test.test_rerank(args, test_ds, model, rerank_bs=args.rerank_batch_size,
                                                num_local=args.num_local,
                                                rerank_dim=(args.local_dim + 3), reg_top=args.reg_top)
        logging.info(f"Recalls on test set {test_ds}: {recalls_str}")
        return 0

    #### Training loop
    for epoch_num in range(start_epoch_num, args.epochs_num):
        if args.optim == 'adamw':
            adjust_learning_rate(optimizer, epoch_num, args)
        for param_group in optimizer.param_groups:
            writer.add_scalar("lr_rate", param_group['lr'], epoch_num)
        # break
        logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)

        # How many loops should an epoch last (default is 5000/1000=5)
        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            # break
            logging.debug(f"Cache: {loop_num} / {loops_num}")

            # Compute triplets to use in the triplet loss
            if (args.cache_refresh_rate != args.queries_per_epoch) or epoch_num == 0:
                triplets_ds.is_inference = True
                triplets_ds.compute_triplets(args, model)
                triplets_ds.is_inference = False

            triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                     batch_size=args.train_batch_size, collate_fn=datasets_ws.collate_fn,
                                     pin_memory=(args.device == "cuda"), drop_last=True)
            if args.fix:
                model = model.eval()
            else:
                model = model.train()
            Reranker = Reranker.train()

            # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=480, W=640
            # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
            for images, triplets_local_indexes, _, utms in tqdm(triplets_dl, ncols=100):
                # Flip all triplets or none
                if args.horizontal_flip:
                    images = transforms.RandomHorizontalFlip()(images)

                # Compute features of all images (images contains queries, positives and negatives)
                features, rerank_features = model(images.to(args.device))  # 全局特征与局部特征
                loss_triplet = 0
                if args.rerank_model == "GCNRerank":
                    features_gcn = features.view(args.train_batch_size, -1, args.fc_output_dim)
                    rerank_features = rerank_features.view(args.train_batch_size, -1, args.num_local,
                                                           args.local_dim + 3)
                    query_features_global = features_gcn[:, 0, :]
                    candinate_feature_global = features_gcn[:, 1:, :]
                    rerank_out, final_out = Reranker(query_features_global,
                                                     candinate_feature_global,
                                                     rerank_features)


                    # (2,11,2),(2,11)
                if args.criterion == "triplet":
                    triplets_local_indexes = torch.transpose(
                        triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1,
                        0)  # (10,1,3) ->((0,1,2),(0,1,3),(0,1,4),(0,1,5)......(12,13,14),(12,13,15),(12,13,16......))
                    # neg_index = 1
                    global_num=0
                    for id_t, triplets in enumerate(triplets_local_indexes):
                        queries_indexes, positives_indexes, negatives_indexes = triplets.T
                        # (0,12) (1,13) (2,14)
                        if not args.fix:
                            global_loss = criterion_triplet(features[queries_indexes],
                                                            features[positives_indexes],
                                                            features[negatives_indexes])
                            loss_triplet += global_loss
                            if len(epoch_losses) < 2:
                                logging.debug(
                                    'global loss:{}'.format(
                                        global_loss / (args.train_batch_size * args.negs_num_per_query)))
                        # print("queries_indexes.shape {}, positives_indexes.shape {}, negatives_indexes.shape {}".format(
                        #     queries_indexes.shape, positives_indexes.shape, negatives_indexes.shape))
                        if args.rerank_model == "r2former":
                            rerank_out_pos, final_out_pos = Reranker(features[queries_indexes],
                                                                     rerank_features[queries_indexes],
                                                                     features[positives_indexes],
                                                                     rerank_features[positives_indexes])
                            # # (2,2)  ,(2)
                            rerank_out_neg, final_out_neg = Reranker(features[queries_indexes],
                                                                     rerank_features[queries_indexes],
                                                                     features[negatives_indexes],
                                                                     rerank_features[negatives_indexes])
                            # (2,2)
                        elif args.rerank_model == "GCNRerank":
                            rerank_out_pos = rerank_out[:, positives_indexes[0] - 1, :]
                            rerank_out_neg = rerank_out[:, negatives_indexes[0] - 1, :]
                        # ====================================================================================
                        elif args.rerank_model == "global_rerank":
                            if args.num_classes == 2:
                                rerank_out_pos = rerank_out[:, positives_indexes[0] - 1, :]
                                rerank_out_neg = rerank_out[:, negatives_indexes[0] - 1, :]
                            if args.num_classes == 1:
                                rerank_out_pos = rerank_out[:, positives_indexes[0] - 1]
                                rerank_out_neg = rerank_out[:, negatives_indexes[0] - 1]

                        if args.rerank_loss == 'ce':
                            if args.num_classes == 2:
                                target = torch.zeros(rerank_out_pos.shape[0] * 2, dtype=torch.long).to(
                                    args.device)  # +num_pairs
                            elif args.num_classes == 1:
                                target = torch.zeros(rerank_out_pos.shape[0] * 2).to(args.device)  # +num_pairs
                            target[:rerank_out_pos.shape[0]] = 1
                            rerank_loss = CE(torch.cat([rerank_out_pos, rerank_out_neg], dim=0),
                                             target)
                            loss_triplet += rerank_loss  # , rerank_out_mix
                            if len(epoch_losses) < 2:
                                logging.debug('rerank loss:{}'.format(
                                    rerank_loss / (args.train_batch_size * args.negs_num_per_query)))
                        elif args.rerank_loss == 'triplet':
                            loss_triplet += relu(
                                sm(rerank_out_neg)[:, 1] - sm(rerank_out_pos)[:, 1] + args.margin).mean()
                        elif args.rerank_loss == 'infoNCE':
                            target = torch.zeros(rerank_out_pos.shape[0], dtype=torch.long).cuda()
                            loss_triplet += CE(
                                torch.cat([sm(rerank_out_pos)[:, 1:], sm(rerank_out_neg)[:, 1:]], dim=1) / args.temperature,
                                target)
                        elif args.rerank_loss == 'combine':
                            if id_t >= args.negs_num_per_query // 2:
                                target = torch.zeros(rerank_out_pos.shape[0] * 2, dtype=torch.long).cuda()  # +num_pairs
                                target[:rerank_out_pos.shape[0]] = 1
                                loss_triplet += CE(torch.cat([rerank_out_pos, rerank_out_neg], dim=0),
                                                   target)
                            else:
                                loss_triplet += relu(sm(rerank_out_neg)[:, 1] - sm(rerank_out_pos)[:, 1]).mean()
                        else:
                            print('No rerank loss!', args.rerank_loss)
                            raise Exception

                elif args.criterion == 'sare_joint':
                    # sare_joint needs to receive all the negatives at once
                    triplet_index_batch = triplets_local_indexes.view(args.train_batch_size, 10, 3)
                    for batch_triplet_index in triplet_index_batch:
                        q = features[batch_triplet_index[0, 0]].unsqueeze(
                            0)  # obtain query as tensor of shape 1xn_features
                        p = features[batch_triplet_index[0, 1]].unsqueeze(
                            0)  # obtain positive as tensor of shape 1xn_features
                        n = features[batch_triplet_index[:, 2]]  # obtain negatives as tensor of shape 10xn_features
                        loss_triplet += criterion_triplet(q, p, n)
                elif args.criterion == "sare_ind":
                    for triplet in triplets_local_indexes:
                        # triplet is a 1-D tensor with the 3 scalars indexes of the triplet
                        q_i, p_i, n_i = triplet
                        loss_triplet += criterion_triplet(features[q_i:q_i + 1], features[p_i:p_i + 1],
                                                          features[n_i:n_i + 1])

                del features
                loss_triplet /= (args.train_batch_size * args.negs_num_per_query)

                optimizer.zero_grad()

                loss_triplet.backward()
                # nn.utils.clip_grad_norm_(model.parameters(),max_norm=10,norm_type=2)
                optimizer.step()
                # i=0
                # for name,parms in model.named_parameters():
                #     if parms.requires_grad==True:
                #         logging.debug("grad_value {}".format(parms))
                #         i+=1
                #         if i>2:
                #             break

                # Keep track of all losses by appending them to epoch_losses
                batch_loss = loss_triplet.item()
                epoch_losses = np.append(epoch_losses, batch_loss)
                del loss_triplet
            writer.add_scalar('Train/current batch_triplet_loss', batch_loss,
                              ((epoch_num + 1) * loops_num + (loop_num + 1)))
            writer.add_scalar('Train/average epoch_triplet_loss', epoch_losses.mean(),
                              ((epoch_num + 1) * loops_num + (loop_num + 1)))

            logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                          f"current batch triplet loss = {batch_loss:.4f}, " +
                          f"average epoch triplet loss = {epoch_losses.mean():.4f}")

        logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                     f"average epoch triplet loss = {epoch_losses.mean():.4f}")
        mean_loss = epoch_losses.mean()
        writer.add_scalar('Train/AvgLoss', epoch_losses.mean(), epoch_num )
        # Compute recalls on validation set
        if ( epoch_num  > 1) and (epoch_num) % 1 == 0:
            recalls, recalls_str = test.test_rerank(args, val_ds, model, rerank_bs=args.rerank_batch_size,
                                                    num_local=args.num_local, rerank_dim=(args.local_dim + 3),
                                                    reg_top=args.reg_top)
            logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
            for i, n in enumerate(args.recall_values):
                writer.add_scalar('Val/Recall_' + str(n), recalls[i], epoch_num + 1)
            save_best = args.save_best
            save_best_r = args.recall_values[save_best]

            is_best = recalls[save_best] > best_r5

            # Save checkpoint, which contains all training parameters
            util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls,
                                        "best_r5": best_r5,
                                        "not_improved_num": not_improved_num
                                        }, is_best, filename="last_model.pth")

            # If recall@5 did not improve for "many" epochs, stop training
            if is_best:
                logging.info(
                    f"Improved: previous best R@{save_best_r} = {best_r5:.1f}, current R@{save_best_r} = {recalls[save_best]:.1f}")
                best_r5 = recalls[save_best]
                not_improved_num = 0
            else:
                not_improved_num += 1
                logging.info(
                    f"Not improved: {not_improved_num} / {args.patience}: best R@{save_best_r} = {best_r5:.1f}, current R@{save_best_r} = {recalls[save_best]:.1f}")
                if not_improved_num >= args.patience:
                    logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                    break
    if not args.test:
        writer.close()
    logging.info(f"Best R@{save_best_r}: {best_r5:.1f}")
    logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    #### Test best model on test set
    best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
    model.load_state_dict(best_model_state_dict)

    recalls, recalls_str = test.test_rerank(args, test_ds, model, test_method=args.test_method,
                                            rerank_bs=args.rerank_batch_size, num_local=args.num_local,
                                            rerank_dim=(args.local_dim + 3), reg_top=args.reg_top)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup:
        alpha = epoch / args.warmup
        warmup_factor = 0.1 * (1.0 - alpha) + alpha
        lr *= warmup_factor
    else:
        # if epoch<10:
        lr = lr * (0.1** ((epoch-args.warmup ) // 4))

        # else:
        #     lr = lr * (0.5 ** ((epoch+12 ) //6 ))
    # if args.cos:  # cosine lr schedule
    #     if epoch < args.warmup:
    #         alpha = epoch / args.warmup
    #         warmup_factor = 0.1 * (1.0 - alpha) + alpha
    #         lr *= warmup_factor
    #     else:
    #         lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs_num))
    # else:  # stepwise lr schedule
    #     for milestone in args.schedule:
    #         lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('current lr:', lr)


if __name__ == '__main__':
    run_train()