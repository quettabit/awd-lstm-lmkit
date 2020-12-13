from statistics import mean, stdev
from comet_ml import API, APIExperiment
from comet_ml.api import Metric, Parameter
COMET_WORKSPACE='<your_workspace>'
COMET_PROJECT='<your_project>'

def build_query(dataset, scheme, stat_signi='stat_signi', c=0.0, k=2.0):
    save = Parameter('save')
    data = Parameter('data')
    top_model = Parameter('top_model')
    gss_c = Parameter('gs_smax_c')
    gss_k = Parameter('gs_smax_k')
    return ((data == dataset) & (top_model == scheme) &\
                (save.contains(stat_signi)) & (gss_c == c) &\
                (gss_k == k))
            
def get_exps(comet_api, dataset):
    if dataset == 'data/penn':
        gss_c_k = [(-1.5, 2.5)]
        schemes = ['plif_smax', 'mos', 'smax']
        ss_query = ['stat_signi/aac', 'stat_signi/haa', 'stat_signi/gas']
        """
        gss_c_k = [(-1.5, 2.5)]
        schemes = ['smax', 's_smax', 'gs_smax', 'plif_smax', 'mos']
        ss_query = ['stat_signi/aaa', 'stat_signi/aaa', 'stat_signi/aad',\
                    'stat_signi/aab', 'stat_signi/bac']
        """
    else:
        gss_c_k = [(-1.5, 3.0)]
        schemes = ['plif_smax']
        ss_query = ['stat_signi/aac']
        """
        schemes = ['smax', 's_smax', 'gs_smax', 'plif_smax', 'mos']
        ss_query = ['stat_signi/aab', 'stat_signi/aaa', 'stat_signi/aac',\
                     'stat_signi/aab', 'stat_signi/bac']
        """
    exps = []
    for idx, scheme in enumerate(schemes):
        if dataset == 'data/penn':
            if scheme == 'gs_smax':
                for ck in gss_c_k:
                    exps.append(
                        comet_api.query(
                            COMET_WORKSPACE, COMET_PROJECT, 
                            build_query(
                                dataset, scheme, ss_query[idx], ck[0], ck[1]
                            )
                        )
                    )
            else:
                exps.append(
                    comet_api.query(
                        COMET_WORKSPACE, COMET_PROJECT, 
                        build_query(dataset, scheme, ss_query[idx])
                    )
                )
        elif dataset == 'data/wikitext-2':
            if scheme == 'gs_smax':
                for ck in gss_c_k:
                    exps.append(
                        comet_api.query(
                            COMET_WORKSPACE, COMET_PROJECT, 
                            build_query(
                                dataset, scheme, ss_query[idx], ck[0], ck[1]
                            )
                        )
                    )
            else:
                exps.append(
                    comet_api.query(
                        COMET_WORKSPACE, COMET_PROJECT, 
                        build_query(dataset, scheme, ss_query[idx])
                    )
                )
                

    return exps

def parse_metrics_summary(metrics_summary, metric_name, field_name):
    for metrics in metrics_summary:
        if metrics['name'] == metric_name:
            return metrics[field_name]
    return None

def get_best_train_ppl(metrics_summary, train_ppls):
    step = 0
    for metrics in metrics_summary:
        if metrics['name'] == 'valid_ppl':
            step = metrics['stepMin']
    for train_ppl in train_ppls:
        if train_ppl['step'] == step:
            return train_ppl['metricValue']

def populate_metrics(summary, metric_list, metric_name, 
                        metric_stat, api_exp=None):
    if metric_name == 'train_ppl':
        train_ppls = api_exp.get_metrics(metric=metric_name)
        metric_val = get_best_train_ppl(summary, train_ppls)
    else:
        metric_val = parse_metrics_summary(
            summary, metric_name, metric_stat
        )
    if metric_val is not None:
        if metric_name == 'test_log_prob_press':
            metric_list.append(int(float(metric_val)))
        else:
            metric_list.append(float(metric_val))

def get_metrics(exps):
    metrics = {
        'val_ppls': [],
        'test_ppls': [],
        'train_ppls': [],
        'test_log_prob_rank': []
    }
    for api_exps in exps:
        test_ppls = []
        val_ppls = []
        train_ppls = []
        test_lp_ranks = []
        for api_exp in api_exps:
            summary = api_exp.get_metrics_summary()
            populate_metrics(summary, test_ppls, 'test_ppl', 'valueMin')
            populate_metrics(summary, val_ppls, 'valid_ppl', 'valueMin')
            populate_metrics(
                summary, train_ppls, 'train_ppl', None, api_exp
            )
            populate_metrics(
                summary, test_lp_ranks, 'test_log_prob_press', 'valueMax'
            )
        print(test_ppls)           
        metrics['val_ppls'].append(val_ppls)
        metrics['test_ppls'].append(test_ppls)
        metrics['train_ppls'].append(train_ppls)
        metrics['test_log_prob_rank'].append(test_lp_ranks)
        
    return metrics

def mean_and_sd(metrics):
    for k, v in metrics.items():
        print("metric is {} ..".format(k))
        for exp_v in v:
            print("mean {:.2f}, sd {:.2f}".format(mean(exp_v), stdev(exp_v)))

def start():
    comet_api = API(api_key='<your_api_key>')
    exps = get_exps(comet_api, 'data/wikitext-2')
    print('got exps')
    metrics = get_metrics(exps)
    print('got metrics')
    mean_and_sd(metrics)    


if __name__ == '__main__':
    start()
 
