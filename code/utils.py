import re
import unicodedata


def delete_value_from_vector(vector, value):
    '''Delete a given value from a vector.

    To be used only when the value is in the vector.
    '''
    if value in vector:
        vector.remove(value)
        return vector
    else:
        raise ValueError('The asked value is not in the vector.')

def text_to_line(raw_text, empty=True):
    r'''Split a raw text into a list of sentences (string) according to '\n'.'''
    split_text = re.split('\n', raw_text)
    if ('' in split_text) and empty: # To remove empty lines
        return delete_value_from_vector(split_text, '')
    else:
        return split_text

def deduplicate_list(original_list):
    '''Remove duplicates from a list, while keeping the original order.'''
    return list(dict.fromkeys(original_list))


class Text:
    '''Basic processing of a text file.
    
    Parameters
    ----------
    file_path : string
        Path to the file
    empty : bool
        Empty lines are removed if True (default: True)

    Attributes
    ----------
    raw_file : string
        Text as a character string
    split_file : list of strings [string]
        Text converted into a list of sentences
    n_sent : integer
        Number of sentences in the text
    '''
    def __init__(self, file_path, empty=True):
        self.raw_file = open(file_path, 'r').read()
        normalised_text = unicodedata.normalize('NFC', self.raw_file)
        self.pp_file = re.sub(' +', ' ', normalised_text)

        self.split_file = text_to_line(self.pp_file, empty=empty)
        self.n_sent = len(self.split_file)
        print(f'There are {self.n_sent} sentences.')
