#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:30:25 2021

@author: rohankumar
"""

import pandas as pd
import numpy as np
from bokeh.plotting import figure
import plotly.express as px
import pickle

#ML Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from kneed import KneeLocator


#from sklearn import KMeans


import spotipy
import streamlit as st
import SessionState


from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials

# Get the credentials from the environment variables
client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('CLIENT_ID'),
                                                      client_secret=os.environ.get('CLIENT_SECRET'))
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


#%%
st.set_page_config(page_title='Music with Machine Learning', page_icon=":notes:")
session_state = SessionState.get(checkboxed=False, num=2)
#%%

def main():

    st.title("Spotify Music Suited to your Mood!")
    #total_playlist = st.sidebar.selectbox("How many playlist?",("1","2","3"))
    
    # set the URI from Spotify
    pl_uris = playlist_input()
    
    # Set the Username from Spotify
    user_id = user_id_input()
    df = get_dataset(pl_uris, user_id, 0)
    
    data = df.copy()
    
    # Pickle the data to local directory for Google Collab
    with open('data.pickle', 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    
    # Name the playlists 
    if df is not None:
        st.write("Playlists you chose ", set(list(df['playlist'])))
        st.write(f'Playlist has {len(df)} tracks.')
        st.write(df)
        
        
    ## Visualizations
   
    st.markdown('### Distribution of Artists and Songs')
    fig = px.bar(df, y = np.unique(df.artist), x=df.artist.value_counts(),
                 labels={'x':'Count','y':'Artist'})
    st.plotly_chart(fig, use_container_width=True)
    
    std_df = PreProcess(df)
    
    ## PCA
    pca = PCA()
    pca.fit(std_df)
    display_variance(pca)
    
    ## We need cumsum(var)>95% to explain 95% variance
    explained_var = pca.explained_variance_ratio_
    for i, e in enumerate(np.cumsum(explained_var)):
        if e > 0.80:
            components = i + 1
    pca = PCA(n_components=components)
    pca.fit(std_df)
    pca_components = pca.transform(std_df)
    st.write("Number of PCA components are : ", components)
    
    n_clusters = find_optimal_clusters_and_display(pca_components)
    
    df_seg_pca_kmedoids = run_KMedoids(n_clusters, pca_components, data, components)
    
    visualize(df_seg_pca_kmedoids)
    
    
    
    
    
#%%    
def playlist_input():
    defaults = "spotify:playlist:05vLcLsdaxixIp3qteEYDC"
    st.sidebar.write("Copy the URI from the Spotify Playlist. Go to the '...' and click on share playlist and copy the link!")
    pl_uris = st.sidebar.text_input("Playlist URI", defaults)
    return pl_uris

def user_id_input():
    # Get the track from the playlist with given id
    user_id = "mwc69r30dn6laux0e97n1zkfo"
    return user_id

def get_dataset(pl_uris, user_id, offset):
    # Add tracks to the playlist as a dataframe
    nau_df = addTracks(pl_uris, user_id, offset)
    
    # Add the audio features to the above dataframe
    return(audioFeatures(nau_df))
    
#%% Make dataframe with name, artists and uris

def compile(names, artists, tracks, uris, playlist_name, playlist_id):
    nau_df = pd.DataFrame(np.column_stack([names, artists, uris, playlist_name, playlist_id]),
                          columns=['names','artists','uris','playlist_name','playlist_id'])
    # Drop the row for which uri is not of length 36. invalid_uris is a list of all the uris that are not valid.
    invalid_uris = list(filter(lambda x: len(x)!=36 , nau_df['uris']))
    nau_df = nau_df[~nau_df['uris'].isin(invalid_uris)]
    return nau_df
    
#%%
## Function to add all the tracks from the playlist
def addTracks(pl_uri,username,offset):
    tracks = []
    pl_id = pl_uri.split(':')[2]
    pl_name = sp.user_playlist(username, pl_id)['name']
    while True:
        results = sp.user_playlist_tracks(user=None, playlist_id = pl_id, fields=None, limit=100, offset=offset, market=None)
        tracks = tracks + results['items']
        if results['next'] is not None:
            offset += 100
        else:
            break
    playlist_name, playlist_id, names, artists, uris = [], [], [], [], []
    
    # Extract metadata from the tracks such as album name, artitsts and URI
    for track in tracks:
        names.append(track['track']['name'])
        artists.append(track['track']['artists'][0]['name'])
        uris.append(track['track']['uri'])
        playlist_name.append(pl_name)
        playlist_id.append(pl_id)
        
    return(compile(names, artists, tracks, uris, playlist_name, playlist_id))
    

#%% Get the audio features for each track

@st.cache(allow_output_mutation=True)
def audioFeatures(nau_df):
    df = pd.DataFrame(columns=['name', 'artist', 'track_URI', 'acousticness', 'danceability', 'energy',
                           'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'playlist'])
    ## Form a dataframe with all the information about the tracks in the playlist
    for name, artist, track_uri, playlist_name in zip(list(nau_df.names), list(nau_df.artists), list(nau_df.uris), list(nau_df.playlist_name)):
        try:
            audio_features = sp.audio_features(tracks = track_uri)
            selected_features =  [audio_features[0][col] for col in df.columns if col not in ["name", "artist", "track_URI", "playlist"]]
            row = [name, artist, track_uri, *selected_features, playlist_name]
            df.loc[len(df.index)] = row
        except:
            print('Error')
    return df
#%% Data Preprocessing

def PreProcess(df):
    # Drop the non-feature columns
    cols_to_drop = ['name','artist','track_URI','playlist']
    df = df.drop(columns=cols_to_drop)
    
    # Standardize the data 
   
    scaler = StandardScaler()
    std_df = scaler.fit_transform(df)

    return std_df
#%%
def display_variance(pca):
    
    #fig, ax=plt.subplots()
    df=pd.DataFrame()
    x = np.arange(0,9,step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    df['x'] = x
    df['explained_variance'] = y
    fig = px.line(df, x='x', y='explained_variance', title='Explained Variance vs PCA Components')
    #fig1.add_vline(y=0.95, line_width=3, line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)

#%%
def find_optimal_clusters_and_display(pca_components):
    wcss = []
    max_clusters = 21
    for i in range(1, max_clusters):
        kmedoids_pca = KMedoids(n_clusters=i, random_state=0)
        kmedoids_pca.fit(pca_components)
        wcss.append(kmedoids_pca.inertia_)
    n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
    st.write("Optimal number of clusters", n_clusters)
    return n_clusters
    
    

#%%
## KMedoids
def run_KMedoids(n_clusters, pca_components, data, components):
    clustering = KMedoids(n_clusters=7, random_state=0)
    clustering.fit(pca_components)
    df_seg_pca_kmedoids = pd.concat([data.reset_index(drop=True), pd.DataFrame(pca_components)], axis=1)
    df_seg_pca_kmedoids.columns.values[(-1*components):] = ["Component " + str(i+1) for i in range(components)]
    df_seg_pca_kmedoids['Cluster'] = clustering.labels_
    return df_seg_pca_kmedoids

#%%
def visualize(df_seg_pca_kmedoids):
    x = df_seg_pca_kmedoids['Component 2']
    y = df_seg_pca_kmedoids['Component 1']
    fig = px.scatter(df_seg_pca_kmedoids, x=x, y=y, color='Cluster')
    st.plotly_chart(fig, use_container_width=True)
#%%
if __name__ == "__main__":
    main()
    





























