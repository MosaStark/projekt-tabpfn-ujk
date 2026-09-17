import argparse
import base
import exp

def make_ablat(ablat_exp):
    data,splits=ablat_exp.get_data()
    clf_type=ablat_exp.get_clf()
    base.make_dir(ablat_exp.out_path)
    for i in range(data.n_cats()):
        data_i=data.remove_col(i)
        result_i,_=splits( data_i,
                           clf_type)
        out_i=f"{ablat_exp.out_path}/{i}"
        result_i.save(out_i)

def ablat_exp(in_path):
    conf=base.read_json(in_path)
    exp_params=exp.ExpParams( conf["data_path"],
                             conf["split_path"],
                             conf["out_path"],
                             conf["clf"])
    make_ablat(exp_params)
    print(exp_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",type=str,default="ablat_conf.json") 
    args=parser.parse_args()
    ablat_exp(args.conf_path)