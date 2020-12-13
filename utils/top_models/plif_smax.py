import torch

def patch_and_freeze(top_model, args):
    if args.plif_smax_patch != '':
        if torch.cuda.is_available() and args.cuda:
            src_model = torch.load(args.plif_smax_patch)
        else:
            src_model = torch.load(args.plif_smax_patch, map_location='cpu')
        param_sum = top_model.plif_w.data.sum().item()
        print("before patching {:.12f}".format(param_sum))
        top_model.plif_w.data = src_model['top_model.plif_w']
        param_sum = top_model.plif_w.data.sum().item()
        print("after patching {:.12f}".format(param_sum))
    if args.plif_smax_freeze:
        top_model.plif_w.requires_grad = False      
        print("freezing plif_w with its sum {:.12f}".format(
            top_model.plif_w.sum().item()
        ))
    return top_model