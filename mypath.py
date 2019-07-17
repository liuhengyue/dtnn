class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 'dataset/UCF-101'

            # Save preprocess data into output_dir
            output_dir = 'dataset/ucf101'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        elif database == '20bn-jester':
            # folder that contains class labels
            root_dir = 'dataset/20bn-jester'

            output_dir = 'dataset/20bn-jester-preprocessed'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return 'models/ucf101-caffe.pth'
    @staticmethod
    def cpm_model_dir():
        return 'ckpt/model_epoch100.pth'