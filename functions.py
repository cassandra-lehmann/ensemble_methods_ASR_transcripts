

#-------------------------------------------------------------------------------------
#                             EXPLORE JSON FILES                               
#-------------------------------------------------------------------------------------

def get_file_names(path, corpus = ''):
    ''' lists all available file names (for a certan corpus, if corpus is not null).
        input: path  = folder that contains the JSON files with raw data
               corpus = the names of the files to be considered for creating the data frame.
        output: alphabetically sorted file names
    '''
    
    import os 
    files = os.listdir(path)
    try:
        files.remove('read_me.txt')
    except:
        pass
    try:
        files.remove('.ipynb_checkpoints')
    except:
        pass
    if (corpus != ''):
        matching_files = [file for file in files if corpus in file]
        return sorted(matching_files)
    else:
        return sorted(files)

def get_machine_ids(files):
    ''' lists all machines for a certain list of files
        input: files = list of files
        output: alphabetically sorted machine ids
    '''
    machine_ids = []
    for index in range(len(files)):
        machine_ids.append(files[index].split('__')[-1][0])
    machine_ids.sort()
    return set(machine_ids)

def get_configurations(files):
    ''' lists all configurations for a certain list of files
        input: files = list of files
        output: alphabetically sorted configurations
    '''
    configuration_ids = []
    for index in range(len(files)):
        configuration_ids.append(files[index].split('__')[1].split('.')[0])
    configuration_ids.sort()
    return set(configuration_ids)

def get_corpora(files):
    ''' lists all corpora for a certain list of files
        input: files = list of files
        output: alphabetically sorted corpora
    '''
    corpus_ids = []
    for index in range(len(files)):
        corpus_ids.append(files[index].split('__')[0].split('.')[0])
    corpus_ids.sort()
    return set(corpus_ids)

#-------------------------------------------------------------------------------------
#                             CREATE THE RAW DATA FRAME                                
#-------------------------------------------------------------------------------------
def create_df_raw(path, files):
    ''' creates the main data frame with all raw data
        input: path  = the folder that contains the JSON files with raw data
               files = the names of the files to be considered for creating the data frame.
        output: df = data frame with raw data
        note: funtion also prints the file that is being processed
    '''
    import pandas as pd
    import json
    from pandas.io.json import json_normalize
    
    df = pd.DataFrame()
    
    for file in files:
        filename = path + file
        with open(filename, 'r') as f:
            data = json.load(f)
        print(file)
        df_intermediate=pd.DataFrame()

        key = file.replace(".json", "")
        machine = file.split('__')[-1][0]
        corpus = file.split('__')[0]
        configuration = file.split('__')[1].split('.')[0]

        df_intermediate = json_normalize(data[key], max_level=3)
        df_intermediate['corpus'] = corpus
        df_intermediate['machine'] = machine
        df_intermediate['configuration'] = configuration
        df_intermediate['file'] = file

        df = pd.concat([df, df_intermediate], axis=0, sort=False)  
        
    return df

#-------------------------------------------------------------------------------------
#                             VARIOUS CHECKS                               
#-------------------------------------------------------------------------------------
def check_special_characters (df, column):
    ''' checks if a certain column in a data frame contains special characters
        input: df  = analysis data frame
               column = column to be checked for special characters
        output: df_special_characters = the data frame, filtered to show the cases where where column contains special characters
    '''
    import regex as re
    special_char_list = ['!', '"', '#',
                     '$', '%', '&',
                     '(', ')', '*',
                     '+', ',', '-'
                     '.', '/', ':',
                     ';',' <', '=',
                     '>', '?', '@',
                     '[' ,'\\', ']',
                     '^' ,'_', '`',
                     '{', '|', '}',
                     '~']
    pattern = '|'.join(['({})'.format(re.escape(c)) for c in special_char_list])
    df_special_characters = df[df[column].str.contains(pattern)]
    
    return df_special_characters

#-------------------------------------------------------------------------------------
#                             SPLIT REFERENCE TEXTS IN TRAIN, VALIDATION, TEST                            
#-------------------------------------------------------------------------------------

def refereces_train_validation_test (all_references, size_validation, size_test):
    ''' splits the list of reference texts in train, validation, test.
        input: all_references  = list of reference texts
               size_validation, size_test = what percentage of reference texts to be considered for validation/test
        output: reference_texts_train, reference_texts_validation, reference_texts_test = the lists of train, vlaidation, test reference texts
    '''
    import random
    
    nr_refs_for_validation = int(size_validation * len(all_references))   
    nr_refs_for_test = int(size_test * len(all_references))        

    random.seed(42)
    reference_texts_test = random.sample(all_references, nr_refs_for_test)

    random.seed(42)
    reference_texts_validation = random.sample(list(set(all_references) - set(reference_texts_test)), nr_refs_for_validation)

    random.seed(42)
    reference_texts_train = list(set(all_references) - set(reference_texts_validation) - set(reference_texts_test))
    
    return reference_texts_train, reference_texts_validation, reference_texts_test


#-------------------------------------------------------------------------------------
#                             PREPARE FOR ALIGNMENT                             
#-------------------------------------------------------------------------------------

def make_reduced_df (df, machines_keep):
    ''' filters a data frame to contain the data for certain machines.
        input: df  = analysis data frame
               machines_keep = array of machines to keep
        output: df = filtered data frame
    '''
    df = df[df['machine'].isin(machines_keep)]
    return df

def split_hypothesis(df):
    ''' splits the hypothesis texts into list of words
        input: df  = analysis data frame
        output: df = enriched data frame
    '''
    df['texts'] = df['hypothesis.text'].apply(lambda txt: txt.split(' '))
    return df

def get_intermediate_df (df, golden_truth):
    ''' gets those rows from the data frame where the reference text is a certain the golden_truth (i.e. gets all hypothesis for a certain reference text)
        input: df  = analysis data frame
        output: df = filtered data frame
    '''
    df = df[df['reference.text'] == golden_truth].reset_index()
    return df

def build_mutations(ref, hyp, position):
    ''' aligns to text sequences
        input: ref  = reference text, as array of words
               hyp = hypothesis text, as array of words
               position = parameter for pairwise2, indicatin which of the possible alignments to keep
        output: mutations = indicates operations (CORRECT, DELETION, SUBSTITUTION, INSERTION) made when aligning hyp to ref.
                composite = the composite sequence resulted from the alignment of hyp to ref.
        note: documentation for Bio.pairwise2 can be found under https://biopython.org/DIST/docs/api/Bio.pairwise2-module.html
    '''
    
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment
    import numpy as np

    mutations = []
    alignments = pairwise2.align.globalmx(ref,hyp,10,1, gap_char=["_"])[position]
    ref_split = alignments[0]
    hyp_split = alignments[1]
    for element in range(len(ref_split)):
        if(ref_split[element] == hyp_split[element]):
            res = 'CORRECT'
        elif (ref_split[element]=='_'):
            res  ='INSERTION'
        elif (hyp_split[element]=='_'):
            res = 'DELETION'
        else:
            res = 'SUBSTITUTION'
        mutations.append([res, ref_split[element], hyp_split[element]])
    composite = list(np.array(mutations)[:,1])
    return mutations, composite   

def create_wtn (utterances):
    ''' creates a composite sequence by aligning multiple text sequences, two at a time
        input: utterances  = text sequences, as arrays of words
        output: wtn = the composite sequence resulted from the alignment.
        note: the order of the utterances matters
    '''
    import numpy as np
    mut, comp = build_mutations(utterances[0], utterances[1], -1)
    for M in utterances[1:]:
        mut, comp = build_mutations(comp, M, -1)
    wtn = list(np.array(mut)[:,1])
    return wtn

def aling_to_wtn(wtn, utterances, configurations = []):
    ''' aligns multiple utterances to a composite sequence
        input: wtn = composite sequence resulted from the alignment of multiple sequences
               utterances  = text sequences, as arrays of words
               configurations = analysis configurations (can contain REF or not)
        output: aligned = data frame of aligned utterances.
    '''
    import numpy as np
    import pandas as pd
    
    aligned = []
    for M in utterances:
        mut, comp = build_mutations(wtn, M, -1)
        aligned.append(np.array((mut))[:,2])
    aligned = pd.DataFrame(aligned)  
    if(configurations != []):
        aligned = aligned.assign(configuration = configurations.tolist())
    return aligned 

def make_alignment_trasposed (dfr, golden_truth):
    ''' transposes data frame with alignment, having the configuration as index
        input: dfr  = analysis data frame
               golden_truth = reference text
        output: df = aligned data frame
    '''
    df = dfr.copy()
    df['ind'] = df['configuration'].astype('object')
    df = df.drop(columns =['configuration'])
    df = (df.set_index(['ind'])).T
    df['reference.text'] = golden_truth
    return df

#-------------------------------------------------------------------------------------
#                             ALIGNMENT, TO REF & TO BEST                           
#-------------------------------------------------------------------------------------

def alignment (df, references, with_reference = False):
    ''' aligns hypothesis texts to either one hypothesis text (the one of the best machine, in terms of average wer) OR to the reference text
        input: df  = analysis data frame
               references = reference text
               with_reference = indicates whethere the aligment is made to the hypothesis of the best machine or to the reference.
        output: alignment_df = aligned data frame
    '''
    import pandas as pd
    import numpy as np
    
    alignment_df= pd.DataFrame()
    count=0
    for golden_truth in references:
        count += 1
        if (count%1000==0):
            print (count)
        df_intermediate = get_intermediate_df(df, golden_truth)
        #print('golden truth:', golden_truth)
        
        #if align with REF is needed, then add the reference text to the utterances list
        if (with_reference == True):
            #get configurations and insert REF
            configurations_original = df_intermediate['configuration'].astype(str).values
            configurations = np.insert(configurations_original, 0, "REF", axis=0)

            #insert REF at the top for alignment
            split_ref = golden_truth.split(' ')
            utterances = [split_ref] + list(df_intermediate['texts'])
        else:
            utterances = list(df_intermediate['texts'])
            configurations = df_intermediate['configuration'].astype(str).values
            
        # align
        wtn = create_wtn(utterances)
        alignment = aling_to_wtn(wtn, utterances, configurations)
        alignment_T = make_alignment_trasposed (alignment, golden_truth)
        alignment_df= alignment_df.append(alignment_T,ignore_index=True)

        if (with_reference == True):
            #place REF at the end
            columnsTitles = np.insert(configurations_original, len(configurations_original), ["REF", "reference.text"], axis=0)
            alignment_df = alignment_df.reindex(columns=columnsTitles)
        
    return alignment_df

#-------------------------------------------------------------------------------------
#                             VOTING                        
#-------------------------------------------------------------------------------------
def mode_modified(series):
    ''' used for voting to get the majoritary word. If there is a tie between a word and the "_" character, take the word.
        input: series = list of words
        output: the word that appears most frequently, or _.
    '''
    counts = series.value_counts()
    if ((len(counts)>1)  & (counts.index[0]=='_')):
        if (counts.values[0] == counts.values[1]):
            return counts.index[1]
        else:
            return counts.index[0]
    else:
        return counts.index[0]
    
    
def voting(alignments):
    ''' creates the sequence of words resulted from majority voting, for a related set of utterances
        input: alignments = aligned utterances
        output: vote = sequence of words resulted from majority voting 
    '''
    import regex as re
    vote=[]
    for column in alignments.columns.values:
        values = alignments[column]
        vote.append(mode_modified(values))
    vote = ' '.join(word for word in vote)
    vote = vote.replace('_', " ").strip(" ")
    vote = re.sub('\s+',' ',vote)
    return vote


def vote (df, references, configurations):
    ''' applies the majority voting to a data frame with multiple reference texts.
        input: df = analysis data frame
               references = selected reference texts 
               configurations = analysis configurations (can contain REF or not)
        output: voting_results_df = data frame containing the results of majority voting (sequence and WER compared to reference text)
    '''
    import pandas as pd
    from jiwer import wer
    
    count = 0
    voting_results = []
    voting_results_df = pd.DataFrame()
        
    for golden_truth in references:
        count += 1
        if (count%1000==0):
            print (count)
            
        # work with all hypothesis for a certain golden truth
        df_intermediate = get_intermediate_df(df, golden_truth)[configurations].T
        
        #vote and retung the majority voting string
        voting_result = voting(df_intermediate)
        
        # compute the similarity between the golden truth and voting result
        voting_wer = wer(golden_truth, voting_result)
        
        # put result in data frame
        voting_results.append([golden_truth, voting_result, voting_wer])
    
    voting_results_df = pd.DataFrame(voting_results, columns = ['reference.text', 'voting_result', 'voting_wer'])
    return voting_results_df


def make_best_asr_df(df, reference_texts):
    ''' builds data frame indicating the best asr/asrs in terms of WER for certain reference texts.
        input: df = analysis data frame
               reference_texts = selected reference texts 
        output: best_asr_df
    '''
    import pandas as pd
    
    best_asr=[]
    best_asr_df = pd.DataFrame()
    count=0
    for golden_truth in reference_texts: 
        count += 1
        if (count%1000==0):
            print (count)
        df_intermediate = get_intermediate_df(df, golden_truth)
        min_wer = min(df_intermediate['recomputed_wer'])
        best_config = df_intermediate[df_intermediate['recomputed_wer'] == min_wer]['configuration'].to_list()
        best_hyp = df_intermediate[df_intermediate['recomputed_wer'] == min_wer]['hypothesis.text'].to_list()
        best_asr.append([golden_truth, round(min_wer, 2), best_config, best_hyp])
    best_asr_df = pd.DataFrame(best_asr, columns = ['reference.text', 'min_wer', 'best_config', 'best_hyp'])
    return best_asr_df


def stats_voting (assessment_voting, references):
    ''' summary of statistics related to the results of majority voting (how may results were better than/worse than/same as the best hypothesis)
        input: df = analysis data frame
               reference_texts = selected reference texts 
        output: best_asr_df
    '''
    import pandas as pd
    
    assessment_voting_sample = assessment_voting[assessment_voting['reference.text'].isin(references)]
    
    s = assessment_voting_sample['voting_class']
    counts = s.value_counts()
    percent = s.value_counts(normalize=True)
    percent100 = s.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    stats = pd.DataFrame({'counts': counts, 'per': percent, 'per100': percent100})
    return stats


#-------------------------------------------------------------------------------------
#                                   Encoding words                               
#-------------------------------------------------------------------------------------

def word_encoding(train, test):
    
    '''
    CONVERTING WORDS INTO NUMERICAL VALUE
    
    - converts words into numerical value using LabelEncoder
    - 'dicti' variable includes all english words available in nltk.corpus into LabelEncoder
    le = fitted LabelEncoder
    '''
    
    import pandas as pd
    from sklearn import preprocessing
    
    df = pd.concat([train, test])
    col = df.columns.tolist()
    col.remove('reference.text')
    df[col] = df[col].astype(str)
    
    from nltk.corpus import words
    dicti = words.words()
    words = dicti + df[col].values.ravel('K').tolist()
    le = preprocessing.LabelEncoder()
    
    le.fit(pd.unique(words))
    
    return le, col


#-------------------------------------------------------------------------------------
#                             Training data preparation                               
#-------------------------------------------------------------------------------------

def train_data_v1(train, reference_texts_train, reference_texts_validation):
    
    '''
    PREPARATION OF TRAINING FOR MODEL TRAINING
    
    machines   = names of ASR machine in training data
    match      = addition column to keep track of ASR machine performances
    col_name   = column name of train data dataframe
    train_data = train_data should be unique to test data (contains only validation and train sentences)
    '''
    import pandas as pd
    
    machines = train.columns[:-2].tolist()
    machines = ["m_" + machine for machine in machines]
    match =pd.DataFrame(columns = machines)

    data = pd.concat([train, match], axis=1)
    data = data.fillna(0)
    data.index = data.index.map(int).sort_values()
    data = data.iloc[(data.index)]
    col_name = data.columns

    train_data = data[data['reference.text'].isin(reference_texts_train) | data['reference.text'].isin(reference_texts_validation)]
    
    return train_data, col_name, machines, match


def upsampling(train_data):
    
    '''
    UPSAMPLING TRAIN DATA
    
    less_10  = words that has less than 10 occurance
    *upsampling is done to ensure model learns words that are uncommon
    '''
    
    import pandas as pd
    from tqdm import tqdm
    
    less_10 = train_data['REF'].value_counts().index[train_data['REF'].value_counts()<10].tolist()
    groups = train_data.groupby('REF')
    
    for uniq in tqdm(less_10):
        number = int(groups.get_group(uniq).shape[0])
        if number < 10:
            num_dup = 10 - number
            train_data = pd.concat([train_data, groups.get_group(uniq).sample(num_dup, random_state=1, replace= True)])
    
    return train_data


def machine_tracker(train_data, col_name, machines, match):
    
    '''
    CREATING ADDITIONAL FEATURES (ASR machine performance tracker)
    
    *model learns which ASR machine predicted previous correct word (look back)
    *adds more weight to machines that correctly predicted pervious words
    '''
    
    from tqdm import tqdm
    
    train_data = train_data.drop('reference.text', axis=1)
    
    for row in tqdm(range(len(train_data['REF']))):
        for mecha in range(len(machines)):
            try:
                if train_data[col_name[mecha]][row] == train_data['REF'][row]:
                    train_data[machines[mecha]][row+1] = 1   
            except:
                empty = 1
    
    return train_data


#-------------------------------------------------------------------------------------
#                             Train & Test split                               
#-------------------------------------------------------------------------------------

def train_test_split(train_data):
        
    '''
    CREATING ADDITIONAL FEATURES (ASR machine performance tracker)
    
    *model learns which ASR machine predicted previous correct word (look back)
    *adds more weight to machines that correctly predicted pervious words
    '''
    
    import pandas as pd
    
    col_train_data = train_data.columns.tolist()
    train_data[col_train_data] = train_data[col_train_data].astype(int)
    
    X_l_train = list(train_data.columns)
    X_l_train.remove('REF')
    y_l_train = train_data['REF']

    X_train = train_data[X_l_train].copy()
    y_train = train_data['REF'].copy()
    
    return X_train, y_train


#-------------------------------------------------------------------------------------
#                             Test data preparation                              
#-------------------------------------------------------------------------------------

def test_data_preparation_1(test, reference_texts_test): 
    
    '''
    CHECK IF TEST DATA IS UNIQUE TO TRAIN DATA
    '''
    
    list_test = test['reference.text'].unique().tolist()
    col_test = test.columns.tolist()
    col_test.remove('reference.text')
    test[col_test] = test[col_test].astype(str)
    
    reference_texts_test_reduced = reference_texts_test[:]
    test_reduced = test[test['reference.text'].isin(reference_texts_test_reduced)]
    
    return test_reduced, col_test, reference_texts_test_reduced

def test_data_preparation_2(test_reduced):
    
    '''
    CREATING ADDITIONAL FEATURES FOR TEST DATA
    
    machines_test   = names of ASR machine in training data
    match_test      = addition column to keep track of ASR machine performances
    test_reduced_df = test dataframe
    '''
    
    import pandas as pd
    
    machines_test = test_reduced.columns[:-1].tolist()
    machines_test = ["m_" + machine for machine in machines_test]
    match_test =pd.DataFrame(columns = machines_test)
    
    test_reduced_df = pd.concat([test_reduced, match_test], axis=1)
    test_reduced_df = test_reduced_df.fillna(0).reset_index()
    test_reduced_df = test_reduced_df.drop('index', axis=1)
    test_reduced_df = test_reduced_df[[c for c in test_reduced_df if c not in ['reference.text']] + ['reference.text']]
    
    return test_reduced_df, machines_test, match_test


def first_prediction(test_reduced_df, regr):
    
    '''
    GET FRIST PREDICTION AND PROBABILITY OF PREDICTION
    
    inputs           = first row in test dataframe
    y_pred_test      = first prediction based on first row
    probability_list = list of probabilities (currently contains only 1 value)
    probability      = probability value as integer
    match_test       = addition column to keep track of ASR machine performances
    test_reduced_df  = test dataframe with first rpw containing prediction from model and probability of prediction
    '''
    
    import pandas as pd
    import itertools 
    
    inputs = pd.DataFrame(test_reduced_df.iloc[0, :-1]).transpose()
    y_pred_test = regr.predict(inputs)
    
    probability_list = regr.predict_proba(inputs)
    
    probability = list(itertools.chain(*probability_list))
    probability = (max(probability)*100).astype(int)
    
    test_reduced_df['REF'] = 0
    test_reduced_df['REF'][0] = y_pred_test

    test_reduced_df['probability'] = 0
    test_reduced_df['probability'][0] = probability
    
    return test_reduced_df

def test_predictions(test_reduced_df, machines_test, match_test, regr, col_test):
    
    '''
    GET PREDICTION OF WHOLE TEST DATAFRAME
    
    inputs = current row
    y_pred_test = prediction of current row
    probability_list = list of prediction probabilities
    
    *function iterates through in test dataframe and predict row by row
    *machine tracker is also implimented
    '''
    
    from tqdm import tqdm
    import pandas as pd
    import itertools 
    
    for row in tqdm(range(test_reduced_df.shape[0])):
        for mecha in range(len(machines_test)):
            try:
                if test_reduced_df[col_test[mecha]][row] == test_reduced_df['REF'][row]:
                    test_reduced_df[machines_test[mecha]][row+1] = 1 

                inputs = pd.DataFrame(test_reduced_df.iloc[row+1, :-3]).transpose()
                y_pred_test = regr.predict(inputs)
                test_reduced_df['REF'][row+1] = y_pred_test

                probability_list = regr.predict_proba(inputs)

                probability = list(itertools.chain(*probability_list))
                probability = (max(probability)*100).astype(int)
                test_reduced_df['probability'][row+1] = probability

            except:
                exception = 1

        if row%100 == 0:
            test_reduced_df.to_csv(r'./Test_predictions_final(full)_test.csv')
    
    return test_reduced_df

#-------------------------------------------------------------------------------------
#                       Raw data preparation for comparison                              
#-------------------------------------------------------------------------------------

def raw_data_preparation(ori_df):
    
    '''
    PREPARATION OF RAW DATAFRAME
    
    *only ASR machines in configs list are considered
    '''
    
    col = ['configuration', 'hypothesis.text', 'reference.text', 'scoring.wer', 'recomputed_wer']
    ori_df = ori_df[col]
    configs = ['5_ae.json',
               '5_a.json', 
               '4_iaebglebg.json', 
               '4_iabglbg.json',
               '3_mdagmlg.json',
               '2_kl.json',
               '3_mdagrmlg.json',
               '2_ka.json']
    ori_df = ori_df[ori_df['configuration'].isin(configs)]

    ori_df['configuration'] = [x.replace('.json', '') for x in ori_df['configuration']]
    
    return ori_df

#-------------------------------------------------------------------------------------
#                                 Model assessment                              
#-------------------------------------------------------------------------------------

def assessment_dataframe_creation(ori_df, test_reduced_df, reference_texts_test_reduced, le):
    
    '''
    CREATING ASSESSMENT DATAFRAME TO COMPARE BEST ASR MACHINE TO MODEL PREDICTION
    
    truth = reference text (ground truth)
    best_machine = name of best ASR machines
    best_machine_sentence = best ASR machine sentence prediction
    model_prediction = model sentence prediction
    model_wer = model word error rate
    machine_wer = machine word error rate
    '''
    
    import regex as re
    import pandas as pd
    import numpy as np
    from jiwer import wer
    from tqdm import tqdm

    ref_test = test_reduced_df['reference.text'].unique().tolist()

    truth = []
    best_machine = []
    best_machine_sentence = []
    model_prediction = []
    model_wer = []
    machine_wer = []

    for sent in tqdm(range(len(ref_test))):
        prediction1 = test_reduced_df[test_reduced_df['reference.text'] == reference_texts_test_reduced[sent]]
        prediction_l_test1 = le.inverse_transform(prediction1['REF']).tolist()
        prediction_l_test1 = ' '.join(prediction_l_test1).replace('_', "")
        prediction_l_test1 = re.sub('\s+', ' ', prediction_l_test1).strip()

        truth.append(reference_texts_test_reduced[sent])
        model_prediction.append(prediction_l_test1 )
        temp_df = ori_df[ori_df['reference.text'] == reference_texts_test_reduced[sent]]
        a = temp_df[temp_df['scoring.wer'] == temp_df['scoring.wer'].min()]['configuration'].tolist()
        b = temp_df[temp_df['scoring.wer'] == temp_df['scoring.wer'].min()]['hypothesis.text'].tolist()[0]
        c = temp_df[temp_df['scoring.wer'] == temp_df['scoring.wer'].min()]['scoring.wer'].tolist()[0]

        best_machine.append(a)
        best_machine_sentence.append(b)
        model_wer.append(wer(reference_texts_test_reduced[sent], prediction_l_test1))
        machine_wer.append(c)

    assessment_model = pd.DataFrame()
    assessment_model['reference.text'] = truth
    assessment_model['best_machine'] = best_machine
    assessment_model['best_machine_sentence'] = best_machine_sentence
    assessment_model['machine_wer'] = machine_wer
    assessment_model['model_hypothesis'] = model_prediction
    assessment_model['model_wer'] = model_wer
        
    return assessment_model


#-------------------------------------------------------------------------------------
#                                 Comparison                              
#------------------------------------------------------------------------------------

def comparison(sent, test_reduced_df, reference_texts_test_reduced, le, assessment_model, ori_df):
    
    '''
    ANALYSIS OF ASR MACHINE AND MODEL PREDICTION 
    
    test_reduced_df = test dataframe
    reference_texts_test_reduced = list of test data sentences
    sent = example sentence choosen from reference_texts_test_reduced list
    *outputs prediction sentences from best ASR machine and model prediction, ground truth, ASR machine WER and model WER
    '''
    
    import jiwer
    from jiwer import wer
    import regex as re
    
    class color:
       PURPLE = '\033[95m'
       CYAN = '\033[96m'
       DARKCYAN = '\033[36m'
       BLUE = '\033[94m'
       GREEN = '\033[92m'
       YELLOW = '\033[93m'
       RED = '\033[91m'
       BOLD = '\033[1m'
       UNDERLINE = '\033[4m'
       END = '\033[0m'
    
    
    prediction1 = test_reduced_df[test_reduced_df['reference.text'] == reference_texts_test_reduced[sent]]
    prediction_l_test1 = le.inverse_transform(prediction1['REF']).tolist()
    prediction_l_test1 = ' '.join(prediction_l_test1).replace('_', "")
    prediction_l_test1 = re.sub('\s+', ' ', prediction_l_test1).strip()

    print(color.BOLD + 'Ground truth: ' + color.END)
    print(color.BLUE + reference_texts_test_reduced[sent] + color.END)
    print('')
    print(color.BOLD +'Model Prediction:'+ color.END)
    print(color.GREEN + prediction_l_test1 + color.END)
    print('')

    temp_df = ori_df[ori_df['reference.text'] == reference_texts_test_reduced[sent]]
    a = temp_df[temp_df['scoring.wer'] == temp_df['scoring.wer'].min()]['configuration'].tolist()[0]
    b = temp_df[temp_df['scoring.wer'] == temp_df['scoring.wer'].min()]['hypothesis.text'].tolist()[0]
    c = temp_df[temp_df['scoring.wer'] == temp_df['scoring.wer'].min()]['recomputed_wer'].tolist()[0]

    print(color.BOLD +'Best machine :'+ color.END, a)
    print(color.RED + b + color.END)
    print('')

    print("-----------------------------------------------------------------------------")
    print(color.BOLD +'WER model:        '+ color.END, wer(reference_texts_test_reduced[sent], prediction_l_test1))
    print(color.BOLD +'WER best machine: '+ color.END, c)
    print("-----------------------------------------------------------------------------")
    print(color.BOLD +'Mean WER model:   '+ color.END, assessment_model['model_wer'].mean())
    print(color.BOLD +'Mean WER model:   '+ color.END, assessment_model['machine_wer'].mean())
    print("-----------------------------------------------------------------------------")
    
    
#-------------------------------------------------------------------------------------
#                             Prediction data preparation                            
#------------------------------------------------------------------------------------- 
    
def prediction_data_preparation(test_df, le, thresh):
    
    '''
    ANALYSIS OF MODEL PREDICTION WITH CONSIDERATION OF PROBABILITY
    SELECTING PREDICTED WORDS BASED ON PROBABILITY THRESHOLD
    
    test_df = dataframe with test prediction and probability
    test_df is transformed into words and '_' is replaces with '' for better interpretability
    
    *function ensures that chosen words (probability sentences) contain the same number of words as model prediction
    '''
    
    from tqdm import tqdm
    
    test_df_transformed = test_df.copy()
    
    transform_columns = ['5_ae', '5_a', '4_iaebglebg', '4_iabglbg', '3_mdagmlg', 
                     '2_kl', '3_mdagrmlg', '2_ka','REF']
    
    for columns in transform_columns:
        test_df_transformed[columns] = le.inverse_transform(test_df_transformed[columns])

    words = []
    test_df_transformed['selected_words'] = None

    test_df_transformed['REF'] = test_df_transformed['REF'].replace('_', '')

    for row in tqdm(range(test_df_transformed.shape[0])):
        if test_df_transformed['probability'][row] > thresh:
            words.append(test_df_transformed['REF'][row])
        else:
            words.append('_')

    test_df_transformed['selected_words'] = words
    test_df_transformed.loc[test_df_transformed['REF'] == '', 'selected_words'] = ''
    
    return test_df_transformed


#-------------------------------------------------------------------------------------
#                             Creating predicted sentences                             
#-------------------------------------------------------------------------------------    

def sentence_probability(sentences, test_df_transformed):
    
    '''
    CREATING SENTENCE BASED ON SELECTED WORDS WITHIN THRESHOLD
    '''
    
    
    from tqdm import tqdm
    import regex as re

    predictions_sentences = []
    selected_sentences = []
    predictions_sentences_num = []

    for sent in tqdm(sentences):
            prediction_word = test_df_transformed[test_df_transformed['reference.text'] == sent]['REF'].tolist()
            prediction_word = ' '.join(prediction_word).replace('_', "")
            prediction_word = re.sub('\s+', ' ', prediction_word).strip()

            selected_word = test_df_transformed[test_df_transformed['reference.text'] == sent]['selected_words'].tolist()
            selected_word = ' '.join(selected_word)#.replace('_', "")
            selected_word = re.sub('\s+', ' ', selected_word).strip()

            predictions_sentences.append(prediction_word)
            selected_sentences.append(selected_word)
            
    return predictions_sentences, selected_sentences


#-------------------------------------------------------------------------------------
#                               Probability Analysis                             
#------------------------------------------------------------------------------------
    
def probability_analysis(sentences, predictions_sentences, selected_sentences, x):
    
    '''
    ANALYSIS OF PROBABILITY SENTENCE AGAINST MODEL PREDICTION AND GROUND TRUTH
    
    outputs ground truth, model prediction sentence, probability sentence and WER of model sentence and probability sentence
    '''
    
    import jiwer
    from jiwer import wer

    class color:
           PURPLE = '\033[95m'
           CYAN = '\033[96m'
           DARKCYAN = '\033[36m'
           BLUE = '\033[94m'
           GREEN = '\033[92m'
           YELLOW = '\033[93m'
           RED = '\033[91m'
           BOLD = '\033[1m'
           UNDERLINE = '\033[4m'
           END = '\033[0m'

    print(color.BOLD + 'Ground truth: ' + color.END)
    print(sentences[x])
    print('')

    print(color.BOLD + 'Model prediction: ' + color.END)
    print(predictions_sentences[x])
    print("-----------------------------------------------------------------------------")
    print(color.BOLD +'WER model:         '+ color.END, wer(sentences[x], predictions_sentences[x]))
    print("-----------------------------------------------------------------------------")

    print('')
    print(color.BOLD + '50% treshold: ' + color.END)
    print(selected_sentences[x])
    print("-----------------------------------------------------------------------------")
    print(color.BOLD +'WER 50% treshold:  '+ color.END, wer(sentences[x], selected_sentences[x]))
    print("-----------------------------------------------------------------------------")
    
       