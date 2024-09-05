import pandas as pd
import configparser
import datetime
from tqdm import tqdm
import json
from openai import OpenAI


def get_five_shots(df,wave_id):
    # Get random five messages from each session
    df_five_shots = df.groupby('sessionId').apply(lambda x: x.sample(5)).reset_index(drop=True)
    df_five_shots = df_five_shots[['sessionId','messageID','timestamp','content','rewritten']]
    # rename content -> original, rewritten -> neutral
    df_five_shots = df_five_shots.rename(columns={'content':'original','rewritten':'neutral','sessionId':'username'})
    df_five_shots.to_csv('data/chat/output/w' +str(wave_id)+ '_five_shot_data.csv')
    return df_five_shots

def get_ten_shots(df,wave_id):
    # Get random ten messages from each session
    df_ten_shots = df.groupby('sessionId').apply(lambda x: x.sample(10)).reset_index(drop=True)
    df_ten_shots = df_ten_shots[['sessionId','messageID','timestamp','content','rewritten']]
    # rename content -> original, rewritten -> neutral
    df_ten_shots = df_ten_shots.rename(columns={'content':'original','rewritten':'neutral','sessionId':'username'})
    df_ten_shots.to_csv('data/chat/output/w' +str(wave_id)+ '_ten_shot_data.csv')
    return df_ten_shots

def get_twenty_shots(df,wave_id):
    # Get random twenty messages from each session
    df_twenty_shots = df.groupby('sessionId').apply(lambda x: x.sample(20)).reset_index(drop=True)
    df_twenty_shots = df_twenty_shots[['sessionId','messageID','timestamp','content','rewritten']]
    # rename content -> original, rewritten -> neutral
    df_twenty_shots = df_twenty_shots.rename(columns={'content':'original','rewritten':'neutral','sessionId':'username'})
    df_twenty_shots.to_csv('data/chat/output/w' +str(wave_id)+ '_twenty_shot_data.csv')
    return df_twenty_shots

def get_all_shots(df,wave_id):
    # Get random twenty messages from each session
    df_all_shots = df[['sessionId','messageID','timestamp','content','rewritten']]
    # rename content -> original, rewritten -> neutral
    df_all_shots = df_all_shots.rename(columns={'content':'original','rewritten':'neutral','sessionId':'username'})
    df_all_shots.to_csv('data/chat/output/w' +str(wave_id)+ '_all_shot_data.csv')
    return df_all_shots




def parse_log_chats(filename):
    df = pd.read_csv(filename, sep='\t', header=None)
    df.columns = ['timestamp', 'log_level', 'message']
    # from the message column, filter only those which json has a field called fromUserId
    df_chats = df[df['message'].str.contains('userId') & df['message'].str.contains('"log"') & df['message'].str.contains('"content')]
    df_chats = df_chats.copy()  # Make a copy to avoid the SettingWithCopyWarning
    df_chats['content'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['content'])
    df_chats['fromUserId'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['userId'])
    df_chats['toUserId'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['to'])
    df_chats['timestamp'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['timestamp'])
    df_chats['sessionId'] = df_chats['message'].apply(lambda x: json.loads(json.loads(x)['log'])['sessionId'])
    df_chats['word_count'] = df_chats['content'].apply(lambda x: len(x.split()))
    # add messageID column to df_chats, first grouping the df_chats by sessionId and then adding a column with the index
    df_chats['messageID'] = df_chats.groupby('sessionId').cumcount()

    return df_chats


def postprocess_and_save_text(df,wave_id):

    # First group by sessionID, then calculate the average_sentence_length and keep only the rows with word_count > average_sentence_length
    df_p = df.copy()
    df_p['average_sentence_length'] = df_p.groupby('sessionId')['word_count'].transform('mean')
    df_p = df_p[df_p['word_count'] > df_p['average_sentence_length']]
    # drop the average_sentence_length column
    df_p = df_p.drop(columns=['average_sentence_length'])

    # save the raw and processed data

    df.to_csv('data/chat/output/w'+str(wave_id)+'_raw_chat.csv', index=False)
    df_p.to_csv('data/chat/output/w'+str(wave_id)+'_processed_chat.csv', index=False)

    return df_p


def create_parallel_corpus(df,wave_id):
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.ini')
    api_key_gpt = config.get('credentials', 'api_key_openai')
    model = "gpt-4o"
    prompt_id = '111'
    prompt_content = '''
    You are an expert in text style transfer. 
    You will be given a sentence written in the conversational style of person X. 
    Your task is to rewrite the same sentence without any style. 
    Here is the sentence written in the style of person X: {}.
    Format result in json as {rewrittenSentence: ""}
    Do NOT provide any additional information or explanation. 
    
    '''

    df['parallelSentence'] = None

    
    final_output = []
    
    for _, row in  tqdm(df.iterrows(),total=df.shape[0],desc = "Creating Parallel Corpus..."):   
        original = row['content']
        query = f"{prompt_content.replace('{}', f'{{{original}}}',1)}"
        
        messages = [{"role": "system", "content": 'You are an linguistics expert.'}, {"role": "user", "content":query}]
        gpt_client = OpenAI(api_key = api_key_gpt)

        chat_response = gpt_client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = 0.2,
            response_format={ "type": "json_object" }

        )
        final_output.append({'timestamp': row['timestamp'],'original': original,'output': chat_response.choices[0].message.content,"model": chat_response.model, "prompt_tokens" : chat_response.usage.prompt_tokens,"completion_tokens" : chat_response.usage.completion_tokens,"object" : chat_response.object, "promptID" : prompt_id})

    df_output = pd.DataFrame(final_output)
    df_output = df_output.reset_index(names='messageID')
    # Apply the function to each row and create new columns
    df_output.to_csv('data/chat/output/w'+str(wave_id)+'_parallel_data.csv')

    return df_output

def merge_parallel_data(df_chats, df_par):
    df_par = df_par[['timestamp', 'rewritten']]
    df_chats = pd.merge(df_par, df_chats, on='timestamp')
    return df_chats

def parse_parallel_data(df_par_sent):
    wronglyParsed = 0
    for index, row in df_par_sent.iterrows():  
        try:
            data = fix_and_parse_json(row['output'])
            # Access the value
            sentence = data["rewrittenSentence"]
            # Add the sentence to the DataFrame, in a new column called 'rewritten'
            df_par_sent.loc[index, 'rewritten'] = sentence
        except ValueError as e:
            print(e)
            df_par_sent.loc[index, 'rewritten'] = row['output']
            wronglyParsed += 1
    print('wronglyParsed: ',wronglyParsed)

    return df_par_sent

def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False
    
def fix_and_parse_json(json_str):
    # Check if the JSON is already valid
    if is_valid_json(json_str):
        return json.loads(json_str)
    
    # Remove extra curly braces and try parsing again
    if json_str.startswith('{{') and json_str.endswith('}}'):
        fixed_str = json_str[1:-1]  # Remove the outermost curly braces
        if is_valid_json(fixed_str):
            return json.loads(fixed_str)
    
    raise ValueError("The JSON string is not valid even after removing extra curly braces! Json_str: ", json_str)
