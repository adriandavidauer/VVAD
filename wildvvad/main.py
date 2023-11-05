from utils.parseUtils import parseUtils
from dataset import dataSet
from utils.downloadUtils import downloadUtils
from utils.kerasUtils import kerasUtils

if __name__ == "__main__":
    dataset = dataSet()
    download = downloadUtils()

    parser_init = parseUtils()
    parser = parser_init.parser

    args = parser.parse_args()

    debug = args.debug if args.debug else False

    if args.option == "download":
        # download.download_and_save_speaking_videos()
        print("download")

    if args.option == "createDataset":
        # dataset.create_vector_dataset_from_videos()
        print("Create data set")

    if args.option == "trainModel":
        print("Training Model")
        raise "End"
        # raise "Function not implemented yet!"
        data = dataset.load_data_set_from_pickles()

        kerasUtilities = kerasUtils()
        X_train, X_test, y_train, y_test = \
            kerasUtilities.train_test_split(dataset=dataset)

        # feed it to model
        model = LAND_LSTM_Model()
        lstm, name = model.build_land_lstm(input_shape=(200, 200))
        lstm.fit(x=X_train, y=y_train)

        lstm.evaluate(x=X_test, y=y_test)
