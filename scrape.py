from cProfile import run
import sys
import json
import csv
from snscrape.base import ScraperException
import snscrape.modules.twitter as st
import multiprocessing
import itertools
import pandas as pd
import datetime
from pathlib import Path
import os
import numpy as np
import datetime as dt

def get_user_tweets(user: str, n_tweets:int):


    #timestamping the search right before we access twitter data
    time_of_search = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    sns_user = st.TwitterUserScraper(user)
    try:
        sns_generator = sns_user.get_items()
    except ScraperException:
        return None
    tweets = itertools.islice(sns_generator, n_tweets)

    
    # tweet_id_list = []
    # for tweet in tweets:
    #     tweet_id_list.append(tweet.id)
    # raw_mentions = map(lambda x: x.mentionedUsers, tweets)

    #fetching content as well
    tweet_ids, raw_mentions, tweet_contents_lst = [], [], []

    #expanding the retrieval of data from the generator as calling it
    #repeatedly exhausts it and we are losing data 
    for t in tweets:
        raw_mentions.append(t.mentionedUsers)
        tweet_ids.append(t.id)


        #tw_dict is for text
        tw_dict = {}
        try:
            tw_dict['tweet_id'] = t.id
            tw_dict['datetime'] = t.date
            tw_dict['display_name'] = t.date
        except (ScraperException, AttributeError):
            tw_dict['tweet_id'] = np.nan
            tw_dict['tweet_text'] = np.NaN
            tw_dict['datetime'] = np.NaN
        try:
            tw_dict['tweet_text'] = t.content
        except (ScraperException, AttributeError):
            tw_dict['tweet_text'] = np.NaN            
        
        tw_dict['User_id'] = np.NaN
        tweet_contents_lst.append(tw_dict)

    # tweet_ids = list(map(lambda x: x.id, tweets))

    mentions = []
    try:
        for i in raw_mentions:
            if i and i[0]:
                mentions.append(i[0].username)
    except ScraperException:
        return None
    try:
        user_result = sns_user._get_entity()

    except KeyError:
        user_info = {"potentially_banned": True}
    else:
        if user_result:
            user_info = {"id": user_result.id,
                         "display name": user_result.displayname,
                         "description": user_result.description,
                         "verified": user_result.verified,
                         "potentially_banned": False,
                         "#followers": user_result.followersCount,
                         "#posts": user_result.statusesCount,
                         "#friends": user_result.friendsCount,
                         "#favourites": user_result.favouritesCount,
                         "location": user_result.location,
                         #adding a timestamp as well so that a researcher can identify the the metadata at time x
                         "time_of_search" : time_of_search,
                         }

            for sub_d in tweet_contents_lst:
                sub_d['User_id'] = user_result.id
        else:
            user_info = {"potentially_banned": True}

    for sub_d in tweet_contents_lst:
        sub_d['display_name'] = user
    


    return (tweet_ids, mentions, user_info,  tweet_contents_lst)




def iteration(args):
    """
    gets tweets ids and mentions from one user
    adds mentions to new_users
    goes through all tweets and adds all mentions from there to new_users
    returns the set of new users

    user: name of the user

    Returns: set of all newly found users
    """

    user, n_tweets = args[0], args[1]
    dict_update = get_user_tweets(user, n_tweets)

    if not dict_update:
        return None

    new_users = dict_update[1]
    tweet_contents = dict_update[3]

    return (new_users, (user, dict_update), tweet_contents)


def start_from_user(user: str, max_it: int = 1, n_tweets: int = 100):

    user_dict = {}

    visited_users = set(user)
    
    result = iteration((user, n_tweets))
    new_users = set(result[0])
    user_dict[user] = result[1][1]
    users_to_update_with = set()
    # tweet_contents
    tweet_dict = {"tweet_id" : [],
                  "datetime" : [],
                  "display_name" : [],
                  "tweet_text" : [],
                  "User_id" : []}
    for tweet in result[2]:
        for key, value in tweet.items():
            tweet_dict[key].append(value)
                  
                  
            
            

    

    i = 0
    pool = multiprocessing.Pool(processes=12)
    while i < max_it:

        i += 1

        new_users = new_users.difference(visited_users)

        data = [(sub_user, n_tweets) for sub_user in new_users if sub_user]

        results = pool.map(iteration, data)

        #get text content
        # other_user_tweet_contents = results[2]

        visited_users.update(new_users)
        
        users_to_update_with = filter(None, map(lambda x: x[0] if x else None, results))
        dict_updates = filter(None, map(lambda x: x[1] if x else None, results))

        for user_name, content in dict_updates:
            user_dict[user_name] = content

        new_users = set(
            [user for user_list in users_to_update_with for user in user_list if user])
        users_to_update_with = set()

        # tweet_contents.extend(other_user_tweet_contents)

        for user_result in results:
            if len(user_result) >= 3:
                for tweet in user_result[2]:
                    for key, value in tweet.items():
                        tweet_dict[key].append(value)

        

    return user_dict, tweet_dict


def main(start_user:str, depth:int, num_tweets:int, project_name:str='Project_name', save:bool=True)->tuple:
    """Main wrapper for using snscrape; retrieves data, stores the edge and user data 
    inside dataframes and jsons. Returns a tuple with the folder path to where the 
    data was saved and the dataframe itself

    Args:
        start_user (str): _description_
        depth (int): _description_
        num_tweets (int): _description_
        project_name (str, optional): _description_. Defaults to 'Project_name'.

    Returns:
        tuple: path to output folder, dataframe
    """    

    edges = []
    user_info = {}
    
    global n_tweets
    n_tweets = num_tweets
    print('Getting data via snscrape ...')
    result_dict, tweet_contents = start_from_user(start_user, depth, n_tweets)

    print('Searching from user: ...', start_user)
    for user, content in result_dict.items():
        for mentioned in content[1]:
            print('\tMentioned: ', mentioned)
            edges.append((user,mentioned))


        # # iterate over the tweets and content as well
        # for  tweet_sub_dict in content[3]:
        #     #make sur we have the id as well
        #     tweet_sub_dict['User'] = user
        #     tweet_sub_dict['User_id'] = user_id
        #     tweet_contents_lst.append(tweet_sub_dict)


    edges_set = set(edges)
    out_edges = []

    print('Iterating over edges: ...')
    i=1
    for edge in edges:
        if (edge[1], edge[0]) in edges_set:
            out_edges.append(edge)
            try:
                user_info[edge[0]] = result_dict[edge[0]][2]
                print(i)
                i+=1
            except KeyError:
                continue

    edge_attr_dict = {}
    for edge in out_edges:
        edge_attr_dict[str(edge)] = edges.count(edge)
        

    

    #tweet text content being turned to dataframe and then saved as well
    # tweet_text_dict = {}
    # for tweet in tweet_contents:
    #     tweet_text_dict.setdefault(tweet["display_name"], []).append(tweet)
        
    tweet_text_df = pd.DataFrame(tweet_contents)
    

    #creating the filepath for the outputs
    data_path = Path(f"Data/{project_name}/")
    day = dt.datetime.now().date().isoformat() + "_"
    time = "-".join([str(dt.datetime.now().hour),  str(dt.datetime.now().minute), str(dt.datetime.now().second)])
    

    run_path = data_path / (day + time)
    print('Saving data inside ', run_path)
    os.makedirs(run_path)
    
    run_params_dict = {"start user" : start_user,
                       "recursion depth" : depth,
                       "number of tweets searched per user": n_tweets}

    tweet_text_fpath = str(run_path / 'tweet_text.csv')
    tweet_text_df.to_csv(tweet_text_fpath)

    user_info_fpath = str(run_path / "user_attributes.csv")
    user_info_pd = pd.DataFrame.from_dict(user_info, orient="index")
    user_info_pd.to_csv(user_info_fpath)

    if save:
        save_query_results(run_path, run_params_dict, out_edges, user_info, edge_attr_dict)

    return run_path, run_params_dict, out_edges, user_info, edge_attr_dict

def save_query_results(run_path:str, run_params_dict:dict, out_edges, user_info, edge_attr_dict):

    with open((run_path / 'run_info.json'), 'w') as handle:
        json.dump(run_params_dict, handle)
    
    with open((run_path / 'edge_list.csv'), 'w') as handle:
        writer = csv.writer(handle)
        writer.writerows(out_edges)

    with open((run_path / 'edge_list.json'), 'w') as handle:
        json.dump(out_edges, handle)

    with open((run_path / 'user_attributes.json'), 'w') as handle:
        json.dump(user_info, handle)

    with open((run_path / 'edge_attributes.json'), 'w') as handle:
        json.dump(edge_attr_dict, handle)




if __name__ == "__main__":


    start_user = sys.argv[1]
    depth = int(sys.argv[2])
    n_tweets = int(sys.argv[3])

    main(start_user, depth, n_tweets)

