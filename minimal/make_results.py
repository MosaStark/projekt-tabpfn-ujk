import os.path
import argparse
import base,clf,dataset,nn


def make_splits(data_path, n_repeats, n_splits, split_path):
    base.make_dir(split_path)
    for id_i, path_i in base.iter_files(data_path):
        print(path_i)
        data_i = dataset.read_csv(path_i)
        splits_i = base.SplitGroup.make(data_i,
                                         n_repeats=n_repeats,
                                         n_splits=n_splits)
        out_i = f"{split_path}/{id_i}"
        splits_i.save(out_i)


def make_result(data_path, split_path, result_path, clf_type):
    base.make_dir(result_path)
    for id_i, path_i in base.iter_files(data_path):
        print(path_i)
        data_i = dataset.read_csv(path_i)
        splits_i = base.SplitGroup.read(f"{split_path}/{id_i}")

        out_i = f"{result_path}/{id_i}"
        base.make_dir(out_i)
        out_ij = f"{out_i}/{clf_type.NAME}"
        base.make_dir(out_ij)

        for j, split_j in enumerate(splits_i.splits):
            result_file = f"{out_ij}/{j}.npz"
            if os.path.exists(result_file):
                print(f"  skip {result_file}")
                continue
            clf_j = clf_type()
            split_j.fit_clf(data_i, clf_j)
            result_j = split_j.pred(data_i, clf_j)
            result_j.save(f"{out_ij}/{j}")
            print(f"  saved {result_file}  acc={result_j.get_acc():.4f}")


# mapowanie nazw z linii poleceń na klasy klasyfikatorów
CLF_DICT = {
#    "MLP": nn.MLP,
    "TabPFN":nn.TabPFN,
    "RF": clf.RF,
    "GRAD": clf.GRAD,
    "LR": clf.LR,
    "SVM": clf.SVM,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="zbiory_danych")
    parser.add_argument("--split_path", type=str, default="splits")
    parser.add_argument("--result_path", type=str, default="results")
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--clfs", type=str, nargs="+",
                         default=["TabPFN"],#["LR", "SVM", "RF", "GRAD", "MLP"],
                         help=f"lista klasyfikatorow sposrod: {list(CLF_DICT.keys())}")
    parser.add_argument("--skip_splits", action="store_true",
                         help="pomija generowanie splitow (zaklada, ze juz istnieja)")
    args = parser.parse_args()

    if not args.skip_splits:
        print("=== make_splits ===")
        make_splits(args.data_path,
                    n_repeats=args.n_repeats,
                    n_splits=args.n_splits,
                    split_path=args.split_path)

    print("=== make_result ===")
    for clf_name in args.clfs:
        if clf_name not in CLF_DICT:
            print(f"Nieznany klasyfikator: {clf_name}, pomijam")
            continue
        clf_type = CLF_DICT[clf_name]
        print(f"--- {clf_name} ---")
        make_result(args.data_path,
                    split_path=args.split_path,
                    result_path=args.result_path,
                    clf_type=clf_type)