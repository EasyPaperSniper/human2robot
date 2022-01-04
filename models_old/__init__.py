def create_model(args, character_names, dataset):
    if args.model == 'mul_top_mul_ske':
        args.skeleton_info = 'concat'
        import models_old.architecture
        return models_old.architecture.GAN_model(args, character_names, dataset)

    else:
        raise Exception('Unimplemented model')