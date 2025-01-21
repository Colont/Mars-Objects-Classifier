import os
import sys
import pandas as pd
file_name = 'msl-images'

def file_path(file_name):
    '''
    Function finds current directory of user and searches for the folder with dataset
    Parameter: File_name is the name of the folder with dataset
    Returns: File Path
    '''

    # Find directory of python file
    script_directory = os.path.dirname(os.path.realpath(sys.argv[0]))
    file_path = os.path.join(script_directory, file_name)
    return file_path

def collect_data(file_path):
    '''
    Function collects the data in the folder and stores into a multiple dataframes
    Parameter: File_path is the path to the folder with the dataset
    Returns: Dataframes
    '''
    # File names
    vocab_name = 'msl_synset_words-indexed.txt'
    test_name = 'test-calibrated-shuffled.txt'
    train_name = 'train-calibrated-shuffled.txt'
    val_name = 'val-calibrated-shuffled.txt'
    # Find path
    vocab_path = os.path.join(file_path, vocab_name)
    test_path = os.path.join(file_path, test_name)
    train_path = os.path.join(file_path, train_name)
    val_path = os.path.join(file_path, val_name)
    
    class FileProcessing:
        def __init__(self, vocab_path, test_path, train_path, val_path):

            self.vocab_path = vocab_path
            self.test_path = test_path
            self.train_path = train_path
            self.val_path = val_path

        def read_vocab(self, vocab_path):
            '''
            Read in vocab only as term and ID are in different places than others
            Parameter: Vocab_path
            Return: vocab_df
            '''

            # Open vocab file and read in
            with open(vocab_path, 'r', encoding='utf-8') as vocab:
                lines = vocab.readlines()

                lines_clean = []
                data = []
            # Clean text removing then row seperations (\n)
            for text in lines:
                lines = text.replace('\n','')
                lines_clean.append(lines)
            # Split values into numbers and the key words
            for entry in lines_clean:
                parts = entry.split(maxsplit=1)  
                number = int(parts[0]) 
                term = parts[1]        
                data.append([number, term])

            vocab_df = pd.DataFrame(data, columns=["ID", "Term"])

            return vocab_df

        def read_shuffled(self, file_path):
            '''
            Process for all training, testing, and value datasets as ID and Picture are switched
            Parameters: File_path path to datasets
            Return: df
            '''
            # Read in other datasets
            with open(file_path, 'r', encoding='utf-8') as files:
                lines = files.readlines()

                lines_clean = []
                data = []
            # Same steps as above
            for text in lines:
                lines = text.replace('\n','')
                lines_clean.append(lines)

            for entry in lines_clean:
                parts = entry.split(maxsplit=1)  
                # Flip these and rename as these columns are different from vocab
                number = int(parts[1]) 
                picture = parts[0]        
                data.append([picture, number])

            df = pd.DataFrame(data, columns=["Picture", "ID"]) 

            return df
        
        def process_all(self):
            '''
            Process all data into dataframes
            Return: vocab, train, test, and val dataframes
            '''

            vocab_df = self.read_vocab(self.vocab_path)
            train_df = self.read_shuffled(self.train_path)
            test_df = self.read_shuffled(self.test_path)
            val_df = self.read_shuffled(self.val_path)

            return vocab_df, train_df, test_df, val_df
    # Process all file paths into dataframes
    file_processor = FileProcessing(vocab_path, test_path, test_path, val_path)
    vocab_df, train_df, test_df, val_df = file_processor.process_all()

    return vocab_df, train_df, test_df, val_df
def main():
    
    dataset_path = file_path(file_name)
    vocab_df, train_df, test_df, val_df = collect_data(dataset_path)
    
    return vocab_df, train_df, test_df, val_df
if __name__ == "__main__":
    main()
    