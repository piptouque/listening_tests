
from source.data_loaders.example_data_loader import DatasetGenerator
from source.models.example_model import ExampleModel
from source.trainers.example_train import ExampleTrainer
from source.utils.dirs import create_dirs
from source.utils.config import process_config
from source.utils.logger import Logger
from source.utils.utils import get_args


def main():

    # capture the config path from the run arguments
    # then process the json configuration file
    args = get_args()
    config = process_config(args)

    # create the experiments dirs
    create_dirs([config.save.path.summary_dir,
                config.save.path.checkpoint_dir])
    # create tensorflow session
    # sess = tf.Session()
    # create your data generator
    # data = DataGenerator(config)
    train_data, test_data = DatasetGenerator(config)()

    # create an instance of the model you want
    # model = ExampleModel(config)
    model = ExampleModel()
    # create tensorboard logger
    logger = Logger(config.save)
    # create trainer and pass all the previous components to it
    # trainer = ExampleTrainer(sess, model, data, config, logger)
    trainer = ExampleTrainer(config, model, logger, train_data, test_data)
    trainer.train()


if __name__ == '__main__':
    main()
