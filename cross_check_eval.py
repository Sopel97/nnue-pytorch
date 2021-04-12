import argparse
import features
import serialize
import nnue_bin_dataset
import nnue_dataset
import subprocess
import re
import ctypes
from model import NNUE

def read_model(nnue_path, feature_set):
    with open(nnue_path, 'rb') as f:
        reader = serialize.NNUEReader(f, feature_set)
        return reader.model

def make_data_reader(data_path, feature_set):
    return nnue_bin_dataset.NNUEBinData(data_path, feature_set)

def eval_model(model, item):
    us, them, white, black, outcome, score, psqtindex, lsindex = item
    us = us.unsqueeze(dim=0)
    them = them.unsqueeze(dim=0)
    white = white.unsqueeze(dim=0)
    black = black.unsqueeze(dim=0)

    eval = model.forward(us, them, white, black, psqtindex, lsindex).item() * 600.0
    if them[0] > 0.5:
        return -eval
    else:
        return eval

def eval_model_batch(model, batch):
    us, them, white, black, outcome, score, psqtindex, lsindex = batch.contents.get_tensors('cpu')

    evals = [v.item() for v in model.forward(us, them, white, black, psqtindex, lsindex) * 600.0]
    for i in range(len(evals)):
        if them[i] > 0.5:
            evals[i] = -evals[i]
    return evals

re_nnue_eval = re.compile(r'NNUE evaluation:\s*?(-?\d*?\.\d*)')

def compute_basic_eval_stats(evals):
    min_engine_eval = min(evals)
    max_engine_eval = max(evals)
    avg_engine_eval = sum(evals) / len(evals)
    avg_abs_engine_eval = sum(abs(v) for v in evals) / len(evals)

    return min_engine_eval, max_engine_eval, avg_engine_eval, avg_abs_engine_eval

def compute_correlation(engine_evals, model_evals):
    if len(engine_evals) != len(model_evals):
        raise Exception("number of engine evals doesn't match the number of model evals")

    min_engine_eval, max_engine_eval, avg_engine_eval, avg_abs_engine_eval = compute_basic_eval_stats(engine_evals)
    min_model_eval, max_model_eval, avg_model_eval, avg_abs_model_eval = compute_basic_eval_stats(model_evals)

    print('Min engine/model eval: {} / {}'.format(min_engine_eval, min_model_eval))
    print('Max engine/model eval: {} / {}'.format(max_engine_eval, max_model_eval))
    print('Avg engine/model eval: {} / {}'.format(avg_engine_eval, avg_model_eval))
    print('Avg abs engine/model eval: {} / {}'.format(avg_abs_engine_eval, avg_abs_model_eval))

    relative_model_error = sum(abs(model - engine) / (abs(engine)+0.001) for model, engine in zip(model_evals, engine_evals)) / len(engine_evals)
    relative_engine_error = sum(abs(model - engine) / (abs(model)+0.001) for model, engine in zip(model_evals, engine_evals)) / len(engine_evals)
    print('Relative engine error: {}'.format(relative_engine_error))
    print('Relative model error: {}'.format(relative_model_error))
    print('Avg abs difference: {}'.format(sum(abs(model - engine) for model, engine in zip(model_evals, engine_evals)) / len(engine_evals)))

def eval_engine_batch(engine_path, net_path, fens):
    engine = subprocess.Popen([engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    parts = ['uci', 'setoption name EvalFile value {}'.format(net_path)]
    for fen in fens:
        parts.append('position fen {}'.format(fen))
        parts.append('eval')
    parts.append('quit')
    query = '\n'.join(parts)
    out = engine.communicate(input=query)[0]
    evals = re.findall(re_nnue_eval, out)
    return [int(float(v)*208) for v in evals]

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--net", type=str, help="path to a .nnue net")
    parser.add_argument("--engine", type=str, help="path to stockfish")
    parser.add_argument("--data", type=str, help="path to .bin dataset")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint (used instead of nnue for local eval)")
    parser.add_argument("--count", type=int, default=100, help="number of datapoints to process")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    feature_set = features.get_feature_set_from_name(args.features)
    if args.checkpoint:
      model = NNUE.load_from_checkpoint(args.checkpoint, feature_set=feature_set)
    else:
      model = read_model(args.net, feature_set)
    model.eval()
    data_reader = make_data_reader(args.data, feature_set)

    fens = []
    results = (ctypes.c_int*args.count)()
    scores = (ctypes.c_int*args.count)()
    plies = (ctypes.c_int*args.count)()
    model_evals = []
    i = -1
    done = 0
    while done < args.count:
        i += 1

        item = data_reader.get_raw(i)
        board = item[0]
        if board.is_check():
            continue

        fen = board.fen()
        fens.append(fen)
        results[done] = int(item[2])
        scores[done] = int(item[3])
        plies[done] = 1

        done += 1

    arr = (ctypes.c_char_p * len(fens))()
    arr[:] = [fen.encode('utf-8') for fen in fens]
    b = nnue_dataset.get_sparse_batch_from_fens(feature_set.name.encode('utf-8'), len(fens), arr, scores, plies, results)

    model_evals = eval_model_batch(model, b)
    nnue_dataset.destroy_sparse_batch(b)

    engine_evals = eval_engine_batch(args.engine, args.net, fens)
    compute_correlation(engine_evals, model_evals)

if __name__ == '__main__':
    main()
