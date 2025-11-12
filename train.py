import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl

from model.dgl.graph_classifier import GraphClassifier as dgl_model

from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    # Include max_nodes_per_hop and semantic pruning in cache path to avoid using wrong cache
    max_nodes_str = f"_maxnodes_{params.max_nodes_per_hop}" if params.max_nodes_per_hop else ""
    semantic_str = ""
    if hasattr(params, 'use_semantic_pruning') and params.use_semantic_pruning:
        semantic_str = f"_semantic_pruning_sr{params.stage1_ratio}_pw{params.path_weight}_sw{params.semantic_weight}_tss{params.target_subgraph_size}"

    params.db_path = os.path.join(
        params.main_dir,
        f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}{max_nodes_str}{semantic_str}'
    )

    # Check if cache exists, if not create new one
    if not os.path.isdir(params.db_path):
        logging.info(f"Cache not found at {params.db_path}")
        logging.info(f"Generating new subgraph dataset...")
        generate_subgraph_datasets(params)
        logging.info(f"Subgraph dataset generated successfully!")
    else:
        logging.info(f"Using existing cache at {params.db_path}")

    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label

    # DEBUG: Check edge types in first few subgraphs to validate aug_num_rels
    print(f"[DEBUG] Dataset stats: num_rels={params.num_rels}, aug_num_rels={params.aug_num_rels}")
    print(f"[DEBUG] Sample subgraph edge types:")
    for i in range(min(3, len(train))):
        subgraph_pos, _, _, subgraphs_neg, _, _ = train[i]
        pos_edge_types = subgraph_pos.edata['type']
        neg_edge_types = [g.edata['type'] for g in subgraphs_neg]

        print(f"  Subgraph {i}: pos max type={pos_edge_types.max().item() if len(pos_edge_types) > 0 else 0}, "
              f"neg max types={[t.max().item() if len(t) > 0 else 0 for t in neg_edge_types]}")

        # Check if any edge type exceeds aug_num_rels
        if len(pos_edge_types) > 0 and pos_edge_types.max() >= params.aug_num_rels:
            print(f"[ERROR] Positive edge type exceeds aug_num_rels!")
            print(f"  Max edge type: {pos_edge_types.max().item()}, aug_num_rels: {params.aug_num_rels}")
            raise ValueError(f"Edge type {pos_edge_types.max().item()} >= aug_num_rels {params.aug_num_rels}")

        for j, neg_types in enumerate(neg_edge_types):
            if len(neg_types) > 0 and neg_types.max() >= params.aug_num_rels:
                print(f"[ERROR] Negative edge type exceeds aug_num_rels!")
                print(f"  Max edge type: {neg_types.max().item()}, aug_num_rels: {params.aug_num_rels}")
                raise ValueError(f"Edge type {neg_types.max().item()} >= aug_num_rels {params.aug_num_rels}")

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    valid_evaluator = Evaluator(params, graph_classifier, valid)

    trainer = Trainer(params, graph_classifier, train, valid_evaluator)

    logging.info('Starting training with full batch...')

    trainer.train()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")

    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=100,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=455,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=3,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')

    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')

    # Graph pooling params
    parser.add_argument('--pool_type', '-pool', type=str, default='mean',
                        choices=['mean', 'sum', 'max', 'attention', 'query_attention'],
                        help='graph pooling strategy: mean, sum, max, attention (global), query_attention (head/tail conditioned)')
    parser.add_argument('--pool_heads', '-ph', type=int, default=1,
                        help='number of attention heads for attention pooling (default: 1, recommended: 4 for multi-head)')
    parser.add_argument('--pool_dropout', '-pd', type=float, default=0.0,
                        help='dropout rate for attention pooling (default: 0.0, recommended: 0.1-0.2 for attention)')

    # Semantic pruning params
    parser.add_argument('--use_semantic_pruning', '-sp', action='store_true',
                        help='enable Two-Stage Semantic Pruning for subgraph extraction')
    parser.add_argument('--semantic_embeddings_path', '-sep', type=str, default=None,
                        help='path to pre-trained semantic embeddings (e.g., TransE entity embeddings)')
    parser.add_argument('--stage1_ratio', '-sr', type=int, default=10,
                        help='ratio for Stage 1 pruning (target_M * stage1_ratio nodes kept after Stage 1)')
    parser.add_argument('--path_weight', '-pw', type=float, default=0.6,
                        help='weight for path length scores in Stage 2 (alpha)')
    parser.add_argument('--semantic_weight', '-sw', type=float, default=0.4,
                        help='weight for semantic similarity scores in Stage 2 (beta)')
    parser.add_argument('--target_subgraph_size', '-tss', type=int, default=1000,
                        help='target subgraph size after pruning (M)')


    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)
