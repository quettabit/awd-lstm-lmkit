def add_word_similarity_args(parser):
    parser.add_argument('--model_dataset', type=str,
                        help='path to dataset on which the model was trained')
    parser.add_argument('--similarity_data_dir', type=str)
    
