#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues June 23 10:40:49 2020

@author: sean
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import matplotlib as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import dash_cytoscape as cyto
import datetime as dt
import json
import statistics as stats
import math
from sklearn.preprocessing import StandardScaler
from dash.dependencies import Input, Output, State
from datetime import date

from app import app

fig = go.Figure()

# def remove_whitespace(df):
#     names = list(df.columns)
#     for i in names:
#         if df[i].dtype == np.object:
#             df[i] = df[i].str.strip()
#     return df

# def to_lowercase(df):
#     names = list(df.columns)
#     for i in names:
#         if df[i].dtype == np.object:
#             df[i] = df[i].str.lower()
#     return df


# def clean_data(btn1, vals, data):
#     if vals is not None:
#         if len(vals) > 0 and btn1 != 0:
#             selection = vals
#             #selected = list(selection.values())
#             total = sum(selection)
#             df = pd.DataFrame.from_dict(data)
#             if total == 1:
#                 clean_df = remove_whitespace(df)
#                 clean_dict = clean_df.to_dict()
#                 return clean_dict
#             elif total == 2:
#                 clean_df = to_lowercase(df)
#                 clean_dict = clean_df.to_dict()
#                 return clean_dict
#             elif total == 3:
#                 clean_df = remove_whitespace(df)
#                 clean_df = to_lowercase(clean_df)
#                 clean_dict = clean_df.to_dict()
#                 return clean_dict

# data = pd.read_csv("assets/data/test_data.csv", low_memory=False)
# clean_dict = clean_data(1,list([1,2]),data)

# def preprocessing(uploaded_data):
#     #dictionary to dataframe
#     df = pd.DataFrame(uploaded_data)
#     #todays date in given format
#     today = dt.datetime.utcnow().strftime("%d/%m/%Y")
#     #replace missing discharge dates with today's date
#     df.iloc[:,2].replace(np.nan, today, inplace=True)
#     #replace all missing values with string None
#     df.replace(np.nan, "None", inplace=True)
#     # Remove rows without a referral date
#     df = df[df.iloc[:,1] != "None"]
#     # Convert dates to datetime format
#     df.iloc[:,1] = pd.to_datetime(df.iloc[:,1], format="%d/%m/%Y")
#     df.iloc[:,2] = pd.to_datetime(df.iloc[:,2], format="%d/%m/%Y")
#     # Calculate length of stay for all rows
#     df['LoSdays'] = (df.iloc[:,2] - df.iloc[:,1]).astype('timedelta64[D]')
#     # Remove rows with a negative length of stay
#     df = df[df.LoSdays >= 0]
#     # Sort the data by Client ID then by date
#     colZero = df.columns[0]
#     colOne = df.columns[1]
#     df = df.sort_values([colZero, colOne], ascending=[True, True])
#     # Amalgamate out of area services into a single category
#     # df_ooa = df.copy(deep=True)
#     # wardteam_np = df_ooa.iloc[:,3].values
#     # setting_np = df_ooa.iloc[:,7].values
#     # mask = setting_np =='ooa'
#     # wardteam_np[mask] = str('all ooa services')
#     # del df_ooa['WardTeam']
#     # df_ooa['Team'] = wardteam_np
    
#     # Transform categorical data columns to category type
#     df_cat = df.copy(deep=True)
#     df_cat['TeamCategory'] = df_cat.iloc[:,3].astype('category')
#     df_cat['TeamCode'] = df_cat.iloc[:,10].cat.codes
    
#     return df_cat

# def create_adjacency_matrix(df_cat):
#     ## Create the adjacency matrix
#     n_teams = max(df_cat.iloc[:,11]) + 1
#     servMove = np.zeros((n_teams,n_teams))
#     singles = np.zeros((1))
    
#     uniId = df_cat.iloc[:,0].unique()
    
#     for i in uniId:
#         mask = df_cat.iloc[:,0] == i
#         team_mask = df_cat[mask].iloc[:,11]
#         n_services = len(team_mask)
#         if (n_services > 1):
#             for j in range(0, (n_services - 1)):
#                 servMove[int(team_mask.iloc[j]), int(team_mask.iloc[j+1])] +=1
#             else:
#                 singles = np.vstack((singles,i))
#     #np.savetxt('assets/data/servMove_matrix.csv',servMove, delimiter=",")
    
#     return servMove, singles

# def create_edge_list(servMove):
#     ## Create the edge list
#     edges = np.zeros((1,3))
#     lenRow = servMove.shape[0]
#     for i in range (0,lenRow):           
#         for j in range(0,lenRow):
#             if (int(servMove[j, i]) > 0):
#                 rowData = np.array([[j,i, int(servMove[j, i])]])
#                 edges = np.vstack((edges,rowData))
#     edges = edges.astype(int)
#     edges = edges[1:edges.shape[0], :]
#     lenEdge = edges.shape[0]
    
#     edgeType = np.repeat("Directed", lenEdge)
#     edgeid = np.arange(0, lenEdge)
#     edges = np.vstack((edges[:,0], edges[:,1], edgeType, edgeid, edges[:,2]))
#     edges = np.transpose(edges)
#     edgesdf = pd.DataFrame(edges, columns = ['Source', 'Target', 'Type', 'Id', 'Weight'])
#     edgesdf['Source'] = edgesdf['Source'].astype('int')
#     edgesdf['Target'] = edgesdf['Target'].astype('int')
#     #edgesdf['Weight'].astype('int')
#     #edgesdf.to_csv('assets/data/edge_list.csv', sep=',', index=False)
    
#     return edgesdf
            
# def create_node_list(df_cat):
#     ## Create the node list
#     losMean = df_cat.groupby('TeamCode')['LoSdays'].mean()
#     losMedian = df_cat.groupby('TeamCode')['LoSdays'].median()
    
#     ndf = pd.DataFrame()
#     ndf['TeamCategory'] = df_cat.iloc[:,10]
#     ndf['TeamCode'] = df_cat.iloc[:,11]
#     ndf['Setting'] = df_cat.iloc[:,7]
    
#     df_uni = ndf.drop_duplicates(subset = ['TeamCategory', 'TeamCode', 'Setting'])
#     df_uni.sort_values('TeamCode', inplace=True)
    
#     nodes = np.vstack((df_uni.TeamCode, df_uni.TeamCategory, 
#                           losMean, losMedian, df_uni.Setting))
#     nodes = np.transpose(nodes)
#     nodesdf = pd.DataFrame(nodes,columns = ['ID', 'Label', 'MeanLoS', 'MedianLoS', 'Setting'])
#     #nodesdf.to_csv('assets/data/node_list.csv', sep=',', index=False)
    
#     return nodesdf

# def create_nx_graph(nodeData,edgeData):
#     ## Initiate the graph object
#     G = nx.DiGraph()
    
#     ## Tranform the data into the correct format for use with NetworkX
#     # Node tuples (ID, dict of attributes)
#     idList = nodeData['ID'].tolist()
#     labels =  pd.DataFrame(nodeData['Label'])
#     labelDicts = labels.to_dict(orient='records')
#     nodeTuples = [tuple(r) for r in zip(idList,labelDicts)]
    
#     # Edge tuples (Source, Target, dict of attributes)
#     sourceList = edgeData['Source'].tolist()
#     sourceList = list(map(int, sourceList))
#     targetList = edgeData['Target'].tolist()
#     targetList = list(map(int, targetList))
#     weights = pd.DataFrame(edgeData['Weight']).astype('int')
#     weightDicts = weights.to_dict(orient='records')
#     edgeTuples = [tuple(r) for r in zip(sourceList,targetList,weightDicts)]
    
#     ## Add the nodes and edges to the graph
#     G.add_nodes_from(nodeTuples)
#     G.add_edges_from(edgeTuples)
    
#     return G

# def create_analysis(G,nodes):
#     #Graph metrics
#     g_met_dict = dict()
#     g_met_dict['num_chars'] = G.number_of_nodes()
#     g_met_dict['num_inter'] = G.number_of_edges()
#     g_met_dict['density'] = nx.density(G)
    
#     #Node metrics
#     e_cent = nx.eigenvector_centrality(G)
#     page_rank = nx.pagerank(G)
#     degree = nx.degree(G)
#     between = nx.betweenness_centrality(G)
    
#     # Extract the analysis output and convert to a suitable scale and format
#     e_cent_size = pd.DataFrame.from_dict(e_cent, orient='index',
#                                          columns=['cent_value'])
#     e_cent_size.reset_index(drop=True, inplace=True)
#     #e_cent_size = e_cent_size*100
#     page_rank_size = pd.DataFrame.from_dict(page_rank, orient='index',
#                                             columns=['rank_value'])
#     page_rank_size.reset_index(drop=True, inplace=True)
#     #page_rank_size = page_rank_size*1000
#     degree_list = list(degree)
#     degree_dict = dict(degree_list)
#     degree_size = pd.DataFrame.from_dict(degree_dict, orient='index',
#                                          columns=['deg_value'])
#     degree_size.reset_index(drop=True, inplace=True)
#     g_met_dict['avg_deg'] = degree_size.iloc[:,0].mean()
#     between_size = pd.DataFrame.from_dict(between, orient='index',
#                                           columns=['betw_value'])
#     between_size.reset_index(drop=True, inplace=True)
    
#     dfs = [e_cent_size,page_rank_size,degree_size,between_size]
#     analysis_df = pd.concat(dfs, axis=1)
#     cols = list(analysis_df.columns)
#     an_arr = analysis_df.to_numpy(copy=True)
#     scaler = StandardScaler()
#     an_scaled = scaler.fit_transform(an_arr)
#     an_df = pd.DataFrame(an_scaled)
#     an_st = an_df.copy(deep=True)
#     an_st.columns = cols
#     an_df.columns = cols
#     an_mins = list(an_df.min())
#     for i in range(len(an_mins)):
#         an_df[cols[i]] -= an_mins[i] - 1
#         an_df[cols[i]] *= 6
    
#     colours = ['#8a820f','#e6194B','#f58231','#ffe119','#bfef45','#3cb44b',
#                '#42d4f4','#4363d8','#911eb4','#f032e6']
#     others = ['#c7c3b9']*(len(an_st)-10)
#     all_colours = colours + others
#     names = ['cent_col', 'rank_col', 'deg_col', 'betw_col']
#     names_two = ['cent_lab', 'rank_lab', 'deg_lab', 'betw_lab']
#     names_three = ['cent_id', 'rank_id', 'deg_id', 'betw_id']
#     an_df['id'] = nodes['ID']
#     an_df['lab'] = nodes['Label']
#     top_id = pd.DataFrame()
#     for j in range(len(names)):
#         an_df.sort_values(cols[j], ascending=False, inplace=True)
#         an_df[names[j]] = all_colours
#         top = list(an_df.iloc[0:10,5])
#         top_id[names_three[j]] = list(an_df.iloc[0:10,4])
#         blanks = ['']*(len(an_df)-10)
#         labs = top + blanks
#         an_df[names_two[j]] = labs
#         an_df.sort_index(inplace=True)
    
#     return an_df, an_st, top_id, g_met_dict

# def edge_styling(edges, top_id):
#     colours = ['#8a820f','#e6194B','#f58231','#ffe119','#bfef45','#3cb44b',
#                '#42d4f4','#4363d8','#911eb4','#f032e6']
#     names = ['cent_col', 'rank_col', 'deg_col', 'betw_col']
#     for j in range(len(names)):
#         match_s = pd.DataFrame()
#         match_t = pd.DataFrame()
#         for i in range(len(top_id)):
#             match_s[i] = np.where(edges['Source'] == top_id.iloc[i,j], colours[i],
#                                 '')
#             match_t[i] = np.where(edges['Target'] == top_id.iloc[i,j], colours[i],
#                                 '')
#         edges['col_source'] = match_s[0] + match_s[1] + match_s[2] + match_s[3] + \
#                             match_s[4] + match_s[5] + match_s[6] + match_s[7] + \
#                             match_s[8] + match_s[9]
#         edges['col_target'] = match_t[0] + match_t[1] + match_t[2] + match_t[3] + \
#                             match_t[4] + match_t[5] + match_t[6] + match_t[7] + \
#                             match_t[8] + match_t[9]
#         l = list()
#         for i in range(len(edges)):
#             if edges['col_source'][i] != '':
#                 l.append(edges['col_source'][i])
#             elif edges['col_target'][i] != '':
#                 l.append(edges['col_target'][i])
#             else:
#                 l.append('#c7c3b9')
#         edges[names[j]] = l
    
    
#     return edges

# def net_analysis(nodes_1, edges_1):
#     G_1 = create_nx_graph(nodes_1,edges_1)
#     an_adj, an_st, top_id, g_met_dict  = create_analysis(G_1,nodes_1)
#     #l_data, l_dict = line_plot_data(an_adj)
#     edges_1 = edge_styling(edges_1, top_id)
#     #bar_1 = create_barchart(an_adj,'cent_value','cent_lab','cent_col')
#     #bar_2 = create_barchart(an_adj,'rank_value','rank_lab','rank_col')
#     #bar_3 = create_barchart(an_adj,'deg_value','deg_lab','deg_col')
#     #bar_4 = create_barchart(an_adj,'betw_value','betw_lab','betw_col')
#     #bar_dict = {1: bar_1, 2: bar_2, 3: bar_3, 4: bar_4}
    
#     nodes_list = list()
#     for i in range(len(nodes_1)):
#         c_node = {
#                 "data": {"id": nodes_1.iloc[i,0], 
#                          "e_cent_lab": an_adj.iloc[i,7],
#                          "e_cent": an_adj.iloc[i,0], 
#                          "e_col": an_adj.iloc[i,6],
#                          "rank_lab": an_adj.iloc[i,9],
#                          "rank": an_adj.iloc[i,1], 
#                          "rank_col": an_adj.iloc[i,8],
#                          "deg_lab": an_adj.iloc[i,11],
#                          "deg": an_adj.iloc[i,2], 
#                          "deg_col": an_adj.iloc[i,10],
#                          "betw_lab": an_adj.iloc[i,13],
#                          "betw": an_adj.iloc[i,3], 
#                          "betw_col": an_adj.iloc[i,12]}
                
#             }
#         nodes_list.append(c_node)
    
#     edges_list = list()
#     for j in range(len(edges_1)):
#         c_edge = {
#                 "data": {"source": edges_1.iloc[j,0], 
#                          "target": edges_1.iloc[j,1],
#                          "weight": edges_1.iloc[j,4], 
#                          "color_cent": edges_1.iloc[j,7],
#                          "color_rank": edges_1.iloc[j,8],
#                          "color_deg": edges_1.iloc[j,9],
#                          "color_betw": edges_1.iloc[j,10]}
#             }
#         edges_list.append(c_edge)
    
#     elements = nodes_list + edges_list
    
#     #return bar_dict, elements, g_met_dict, l_data, l_dict, an_adj, top_id
#     return elements, g_met_dict, an_adj, top_id

# df_cat = preprocessing(clean_dict)
# servMove, singles = create_adjacency_matrix(df_cat)
# edgesdf = create_edge_list(servMove)
# nodesdf = create_node_list(df_cat)
# elements, g_met_dict, an_adj, top_id = net_analysis(nodesdf, edgesdf)
# j_met_dict = json.dumps(g_met_dict)

layout = html.Div(children=[
    html.Div(id='home-p3', className='padheader color-section',children=[
        html.Div(className='container',children=[
            html.Div(className='row',children=[
                html.Div(className='col-4 logo',children=[
                    html.A(href='/apps/page1', children=[
                    html.Img(src=app.get_asset_url('images/project_logo_v8.png')),
                    ]),
                ]),
                html.Div(className='col-8 text-center site-title',children=[
                    html.Div(className='site-title-container',children=[
                        html.H1(className='flex-header',children=['Network-based operational modelling platform']),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(id='navbar-p3', children=[
        html.Div(className='padheader color-section', children=[
            html.Div(className='row', children=[
                html.Div(className='container', children=[
                    html.Ul(id='nav-p3', className='nav', children=[
                        html.Li(children=[html.A(children=['Instructions'],href="/apps/page1")]),
                        html.Li(children=[html.A(children=['Data upload'], href="/apps/page2")]),
                        html.Li(children=[html.A(className='active', children=['Network overview'], href="/apps/page3")]),
                        html.Li(children=[html.A(children=['Dynamic network'], href="/apps/page4")]),
                        html.Li(children=[html.A(children=['Reports'], href="/apps/page5")]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(id='page-title-p3', className='padheader color-section', children=[
        html.Div(className='row', children=[
            html.Div(className='container', children=[
                html.H2(className='text-center', children=['Network overview']),
            ]),
        ]),
    ]),
    html.Div(className='padsection', children=[
        html.Div(className='b-container', children=[
            html.H3(children=['Interactive system network map']),
        ]),
        html.Div(className='b-container', children=[
            html.Div(className='row', children=[
                html.Div(className='col-7', children=[
                    html.Div(className='graph-container', children=[
                        cyto.Cytoscape(
                            id='static-net',
                            className='net-obj',
                            elements=list(),
                            style={'width':'100%', 'height':'600px'},
                            layout={'name': 'cose',
                                    'padding': 30,
                                    #'quality': 'proof',
                                    'nodeRepulsion': '7000',
                                    #'gravity': '0.01',
                                    'gravityRange': '6.0',
                                    'nestingFactor': '0.8',
                                    'edgeElasticity': '50',
                                    'idealEdgeLength': '200',
                                    'nodeDimensionsIncludeLabels': 'true',
                                    'numIter': '6000',
                                    },
                            stylesheet=[
                                    {'selector': 'node',
                                     'style': {
                                             'width': 'data(e_cent)',
                                             'height': 'data(e_cent)',
                                             'background-color': 'data(e_col)',
                                             'content': 'data(e_cent_lab)',
                                             'font-size': '40px',
                                             'text-outline-color': 'white',
                                             'text-outline-opacity': '1',
                                             'text-outline-width': '8px',
                                             # 'text-background-color': 'white',
                                             # 'text-background-opacity': '1',
                                             # 'text-background-shape': 'round-rectangle',
                                             # 'text-background-padding': '20px'
                                         }
                                     },
                                    {'selector': 'edge',
                                     'style': {
                                             'line-color': 'data(color_cent)'
                                         }
                                     }
                                ]
                            )
                    ]),
                ]),
                html.Div(className='opt-sec-container', children=[
                    html.Div(className='col-5', children=[
                        html.Div(className='options-container', children=[
                            html.H4('Aggregation and subsetting options'),
                            dcc.Dropdown(id='no-agg',
                            placeholder='Aggregation level',
                            options=[
                                {'label': 'Service', 'value': 'service'},
                                {'label': 'Speciality', 'value': 'speciality'},
                                {'label': 'Locality', 'value': 'locality'},
                                ]),
                            dcc.Dropdown(id='no-att',
                            placeholder='Subset attribute',
                            options=[
                                {'label': 'Service', 'value': 'service'},
                                {'label': 'Speciality', 'value': 'speciality'},
                                {'label': 'Locality', 'value': 'locality'},
                                ]),
                            dcc.Dropdown(id='no-val',
                            placeholder='Subset value',
                            options=[
                                {'label': 'Locality 1', 'value': 'locality_1'},
                                {'label': 'Locality 2', 'value': 'locality_2'},
                                {'label': 'Locality 3', 'value': 'locality_3'},
                                {'label': 'Locality 4', 'value': 'locality_4'},
                                ]),
                        ]),
                        html.Div(className='options-container', children=[
                            html.H4('Network summary'),
                            html.Ul(id='net-sum',children=[
                                html.Li('Number of nodes: '),
                                html.Li('Number of edges: '),
                                html.Li('Graph density: ')
                                ]),
                            html.Button('Create network',id='test-button',n_clicks=0)
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(className='padsection color-section', children=[
        html.Div(className='b-container', children=[
            html.H3(children=['Service SPC charts']),
        ]),
        html.Div(className='b-container', children=[
            html.Div(className='row', children=[
                html.Div(className='col-7', children=[
                    html.Div(className='graph-container', children=[
                        dcc.Graph(id='node-spc',
                                  figure=fig),
                    ]),
                ]),
                html.Div(className='opt-sec-container', children=[
                    html.Div(className='col-5', children=[
                        html.Div(className='options-container', children=[
                            html.H4('Service SPC options'),
                            dcc.Dropdown(id='no-team-spc',
                                         options=list(),
                                         placeholder='Select team/service',
                                         multi=False),
                            dcc.Dropdown(id='no-met-spc',
                                         options=list(),
                                         placeholder='Select metric',
                                         multi=False),
                            html.Div(className='time-opts-container', children=[
                            dcc.DatePickerRange(id='spc-range', className='graph-datepicker',
                                                start_date='2015-06-01',
                                                end_date='2016-01-01',
                                                min_date_allowed=date(1990, 1, 1),
                                                max_date_allowed=date(2021, 12, 30),
                                                initial_visible_month=date(2020, 4, 20)),
                            dcc.Input(id='spc-slice', className='graph-slice',
                                      value='30',
                                      type='number',
                                      placeholder='Time-slice days',
                                      min=1,
                                      max=365,
                                      step=1)
                            ])
                        ]),
                        html.Div(className='options-container', children=[
                            html.H4('Out-of-control signals'),
                            html.Ul(id='spc-sum', children=[
                                html.Li('A single point outside the control limits'),
                                html.Li('Two out of three successive points are on the same side of the centerline and farther than 2 \u03C3 from it'),
                                html.Li('Four out of five successive points are on the same side of the centerline and farther than 1 \u03C3 from it'),
                                html.Li('A run of eight in a row are on the same side of the centerline'),
                                html.Li('Obvious consistent or persistent patterns that suggest something unusual about your data and your process'),
                            ]),
                            html.Ul(id='spc-type')
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(className='padsection', children=[
        html.Div(className='b-container', children=[
            html.H3(children=['Service attribute comparison']),
        ]),
        html.Div(className='b-container', children=[
            html.Div(className='row', children=[
                html.Div(className='col-7', children=[
                    html.Div(className='graph-container', children=[
                        dcc.Graph(id='node-bar',
                                  figure=fig),
                    ]),
                ]),
                html.Div(className='opt-sec-container', children=[
                    html.Div(className='col-5', children=[
                        html.Div(className='options-container', children=[
                            html.H4('Service comparison options'),
                            dcc.Dropdown(id='no-team-comp',
                                         options=list(),
                                         placeholder='Select teams to compare',
                                         multi=True),
                            dcc.Dropdown(id='no-att-comp',
                                         options=list(),
                                         placeholder='Select attribute',
                                         multi=False),
                        ]),
                        html.Div(className='options-container', children=[
                            html.H4('Service comparison summary'),
                            html.Ul(id='no-comp-sum',children=[
                                # html.Li('Mean:'),
                                # html.Li('Median:'),
                                # html.Li('Standard deviation:'),
                                # html.Li('Minimum:'),
                                # html.Li('Maximum:'),
                                # html.Li('Number of nodes:'),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(children=[
        dcc.Store(id='edges-store', storage_type='session'),
        dcc.Store(id='nodes-store', storage_type='session'),
        dcc.Store(id='analysis-df-store', storage_type='session'),
        dcc.Store(id='g-met-store', storage_type='session')
        ]),
    html.Div(id='contact-p3', className='padsection color-section footer-border', children=[
        html.Div(className='b-container', children=[
            html.Div(className='row', children=[
                html.Div(className='col-2', children=[
                    html.H3(children=['Contact details']),
                    html.Div(className='details-container', children=[
                        html.Ul(className='list-unstyled', children=[
                            html.Li(children=[html.Span(children=['Dr Sean Manzi'])]),
                            html.Li(children=[html.Span(children=['South Cloisters'])]),
                            html.Li(children=[html.Span(children=['St Lukes campus'])]),
                            html.Li(children=[html.Span(children=['University of Exeter'])]),
                            html.Li(children=[html.Span(children=['Exeter, Devon'])]),
                            html.Li(children=[html.Span(children=['EX1 2LU'])]),
                            html.Li(children=[html.Span(children=['+44(0)1392 726096'])]),
                        ]),
                    ]),
                ]),
                html.Div(className='col-2', children=[
                    html.Div(className='email-container', children=[
                        html.P(children=['Send an email with any questions or comments to Dr Sean Manzi at:']),
                        html.A(className='contact-email', children=['s.s.manzi@exeter.ac.uk'], href='mailto:s.s.manzi@exeter.ac.uk'),
                    ]),
                ]),
                html.Div(className='col-8', children=[
                    html.Div(className='funding-container text-center', children=[
                        html.P(children=['This project is funded and supported by:']),
                        html.Ul(className='list-unstyled funding', children=[
                            html.Li(children=[html.A(children=[html.Img(src=app.get_asset_url('images/THIS_logo2.png'))], href="https://www.thisinstitute.cam.ac.uk/", target='_blank')]),
                            html.Li(children=[html.A(children=[html.Img(src=app.get_asset_url('images/Exeter_Uni.png'))], href="http://www.exeter.ac.uk/", target='_blank')]),
                            html.Li(children=[html.A(children=[html.Img(src=app.get_asset_url('images/NIHR_Logo.png'))], href="https://www.arc-swp.nihr.ac.uk/", target='_blank')]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(id='licence-p3', className='padsection footer-border', children=[
        html.Div(className='row', children=[
            html.Div(className='b-container text-center', children=[
                html.P(children=['Copyright 2020 Sean Manzi']),
                html.P(children=['Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:']),
                html.P(children=['The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.']),
                html.P(children=['THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.']),
            ]),
        ]),
    ]),
    html.Div(id='hidden-div', style={'display': 'none'})
])

def preprocessing(uploaded_data):
    #dictionary to dataframe
    df = pd.DataFrame(uploaded_data)
    #todays date in given format
    today = dt.datetime.utcnow().strftime("%d/%m/%Y")
    #replace missing discharge dates with today's date
    df.iloc[:,2].replace(np.nan, today, inplace=True)
    #replace all missing values with string None
    df.replace(np.nan, "None", inplace=True)
    # Remove rows without a referral date
    df = df[df.iloc[:,1] != "None"]
    # Convert dates to datetime format
    df.iloc[:,1] = pd.to_datetime(df.iloc[:,1], format="%Y/%m/%d")
    df.iloc[:,2] = pd.to_datetime(df.iloc[:,2], format="%Y/%m/%d")
    df.iloc[:,1] = df.iloc[:,1].dt.strftime("%d/%m/%Y")
    df.iloc[:,2] = df.iloc[:,2].dt.strftime("%d/%m/%Y")
    df.iloc[:,1] = pd.to_datetime(df.iloc[:,1], format="%d/%m/%Y")
    df.iloc[:,2] = pd.to_datetime(df.iloc[:,2], format="%d/%m/%Y")
    # Calculate length of stay for all rows
    df['LoSdays'] = (df.iloc[:,2] - df.iloc[:,1]).astype('timedelta64[D]')
    # Remove rows with a negative length of stay
    df = df[df.LoSdays >= 0]
    # Sort the data by Client ID then by date
    colZero = df.columns[0]
    colOne = df.columns[1]
    df = df.sort_values([colZero, colOne], ascending=[True, True])
    # Amalgamate out of area services into a single category
    # df_ooa = df.copy(deep=True)
    # wardteam_np = df_ooa.iloc[:,3].values
    # setting_np = df_ooa.iloc[:,7].values
    # mask = setting_np =='ooa'
    # wardteam_np[mask] = str('all ooa services')
    # del df_ooa['WardTeam']
    # df_ooa['Team'] = wardteam_np
    
    # Transform categorical data columns to category type
    df_cat = df.copy(deep=True)
    df_cat['TeamCategory'] = df_cat.iloc[:,3].astype('category')
    df_cat['TeamCode'] = df_cat.iloc[:,10].cat.codes
    
    return df_cat

def create_adjacency_matrix(df_cat):
    ## Create the adjacency matrix
    n_teams = max(df_cat.iloc[:,11]) + 1
    servMove = np.zeros((n_teams,n_teams))
    singles = np.zeros((1))
    
    uniId = df_cat.iloc[:,0].unique()
    
    for i in uniId:
        mask = df_cat.iloc[:,0] == i
        team_mask = df_cat[mask].iloc[:,11]
        n_services = len(team_mask)
        if (n_services > 1):
            for j in range(0, (n_services - 1)):
                servMove[int(team_mask.iloc[j]), int(team_mask.iloc[j+1])] +=1
            else:
                singles = np.vstack((singles,i))
    #np.savetxt('assets/data/servMove_matrix.csv',servMove, delimiter=",")
    
    return servMove, singles

def create_edge_list(servMove):
    ## Create the edge list
    edges = np.zeros((1,3))
    lenRow = servMove.shape[0]
    for i in range (0,lenRow):           
        for j in range(0,lenRow):
            if (int(servMove[j, i]) > 0):
                rowData = np.array([[j,i, int(servMove[j, i])]])
                edges = np.vstack((edges,rowData))
    edges = edges.astype(int)
    edges = edges[1:edges.shape[0], :]
    lenEdge = edges.shape[0]
    
    edgeType = np.repeat("Directed", lenEdge)
    edgeid = np.arange(0, lenEdge)
    edges = np.vstack((edges[:,0], edges[:,1], edgeType, edgeid, edges[:,2]))
    edges = np.transpose(edges)
    edgesdf = pd.DataFrame(edges, columns = ['Source', 'Target', 'Type', 'Id', 'Weight'])
    edgesdf['Source'] = edgesdf['Source'].astype('int')
    edgesdf['Target'] = edgesdf['Target'].astype('int')
    #edgesdf['Weight'].astype('int')
    #edgesdf.to_csv('assets/data/edge_list.csv', sep=',', index=False)
    
    return edgesdf
            
def create_node_list(df_cat):
    ## Create the node list
    losMean = df_cat.groupby('TeamCode')['LoSdays'].mean()
    losMedian = df_cat.groupby('TeamCode')['LoSdays'].median()
    
    ndf = pd.DataFrame()
    ndf['TeamCategory'] = df_cat.iloc[:,10]
    ndf['TeamCode'] = df_cat.iloc[:,11]
    ndf['Setting'] = df_cat.iloc[:,7]
    
    df_uni = ndf.drop_duplicates(subset = ['TeamCategory', 'TeamCode', 'Setting'])
    df_uni.sort_values('TeamCode', inplace=True)
    
    nodes = np.vstack((df_uni.TeamCode, df_uni.TeamCategory, 
                          losMean, losMedian, df_uni.Setting))
    nodes = np.transpose(nodes)
    nodesdf = pd.DataFrame(nodes,columns = ['ID', 'Label', 'MeanLoS', 'MedianLoS', 'Setting'])
    #nodesdf.to_csv('assets/data/node_list.csv', sep=',', index=False)
    
    return nodesdf

def create_nx_graph(nodeData,edgeData):
    ## Initiate the graph object
    G = nx.DiGraph()
    
    ## Tranform the data into the correct format for use with NetworkX
    # Node tuples (ID, dict of attributes)
    idList = nodeData['ID'].tolist()
    labels =  pd.DataFrame(nodeData['Label'])
    labelDicts = labels.to_dict(orient='records')
    nodeTuples = [tuple(r) for r in zip(idList,labelDicts)]
    
    # Edge tuples (Source, Target, dict of attributes)
    sourceList = edgeData['Source'].tolist()
    sourceList = list(map(int, sourceList))
    targetList = edgeData['Target'].tolist()
    targetList = list(map(int, targetList))
    weights = pd.DataFrame(edgeData['Weight']).astype('int')
    weightDicts = weights.to_dict(orient='records')
    edgeTuples = [tuple(r) for r in zip(sourceList,targetList,weightDicts)]
    
    ## Add the nodes and edges to the graph
    G.add_nodes_from(nodeTuples)
    G.add_edges_from(edgeTuples)
    
    return G

def create_analysis(G,nodes):
    #Graph metrics
    g_met_dict = dict()
    g_met_dict['num_chars'] = G.number_of_nodes()
    g_met_dict['num_inter'] = G.number_of_edges()
    g_met_dict['density'] = nx.density(G)
    
    #Node metrics
    e_cent = nx.eigenvector_centrality(G)
    page_rank = nx.pagerank(G)
    degree = nx.degree(G)
    between = nx.betweenness_centrality(G)
    
    # Extract the analysis output and convert to a suitable scale and format
    e_cent_size = pd.DataFrame.from_dict(e_cent, orient='index',
                                          columns=['cent_value'])
    e_cent_size.reset_index(drop=True, inplace=True)
    #e_cent_size = e_cent_size*100
    page_rank_size = pd.DataFrame.from_dict(page_rank, orient='index',
                                            columns=['rank_value'])
    page_rank_size.reset_index(drop=True, inplace=True)
    #page_rank_size = page_rank_size*1000
    degree_list = list(degree)
    degree_dict = dict(degree_list)
    degree_size = pd.DataFrame.from_dict(degree_dict, orient='index',
                                          columns=['deg_value'])
    degree_size.reset_index(drop=True, inplace=True)
    g_met_dict['avg_deg'] = degree_size.iloc[:,0].mean()
    between_size = pd.DataFrame.from_dict(between, orient='index',
                                          columns=['betw_value'])
    between_size.reset_index(drop=True, inplace=True)
    
    dfs = [e_cent_size,page_rank_size,degree_size,between_size]
    analysis_df = pd.concat(dfs, axis=1)
    cols = list(analysis_df.columns)
    an_arr = analysis_df.to_numpy(copy=True)
    scaler = StandardScaler()
    an_scaled = scaler.fit_transform(an_arr)
    an_df = pd.DataFrame(an_scaled)
    an_st = an_df.copy(deep=True)
    an_st.columns = cols
    an_df.columns = cols
    an_mins = list(an_df.min())
    for i in range(len(an_mins)):
        an_df[cols[i]] -= an_mins[i] - 1
        an_df[cols[i]] *= 6
    
    colours = ['#8a820f','#e6194B','#f58231','#ffe119','#bfef45','#3cb44b',
                '#42d4f4','#4363d8','#911eb4','#f032e6']
    others = ['#c7c3b9']*(len(an_st)-10)
    all_colours = colours + others
    names = ['cent_col', 'rank_col', 'deg_col', 'betw_col']
    names_two = ['cent_lab', 'rank_lab', 'deg_lab', 'betw_lab']
    names_three = ['cent_id', 'rank_id', 'deg_id', 'betw_id']
    an_df['id'] = nodes['ID']
    an_df['lab'] = nodes['Label']
    top_id = pd.DataFrame()
    for j in range(len(names)):
        an_df.sort_values(cols[j], ascending=False, inplace=True)
        an_df[names[j]] = all_colours
        top = list(an_df.iloc[0:10,5])
        top_id[names_three[j]] = list(an_df.iloc[0:10,4])
        blanks = ['']*(len(an_df)-10)
        labs = top + blanks
        an_df[names_two[j]] = labs
        an_df.sort_index(inplace=True)
    
    return an_df, an_st, top_id, g_met_dict

def edge_styling(edges, top_id):
    colours = ['#8a820f','#e6194B','#f58231','#ffe119','#bfef45','#3cb44b',
                '#42d4f4','#4363d8','#911eb4','#f032e6']
    names = ['cent_col', 'rank_col', 'deg_col', 'betw_col']
    for j in range(len(names)):
        match_s = pd.DataFrame()
        match_t = pd.DataFrame()
        for i in range(len(top_id)):
            match_s[i] = np.where(edges['Source'] == top_id.iloc[i,j], colours[i],
                                '')
            match_t[i] = np.where(edges['Target'] == top_id.iloc[i,j], colours[i],
                                '')
        edges['col_source'] = match_s[0] + match_s[1] + match_s[2] + match_s[3] + \
                            match_s[4] + match_s[5] + match_s[6] + match_s[7] + \
                            match_s[8] + match_s[9]
        edges['col_target'] = match_t[0] + match_t[1] + match_t[2] + match_t[3] + \
                            match_t[4] + match_t[5] + match_t[6] + match_t[7] + \
                            match_t[8] + match_t[9]
        l = list()
        for i in range(len(edges)):
            if edges['col_source'][i] != '':
                l.append(edges['col_source'][i])
            elif edges['col_target'][i] != '':
                l.append(edges['col_target'][i])
            else:
                l.append('#c7c3b9')
        edges[names[j]] = l
    
    
    return edges

def net_analysis(nodes_1, edges_1):
    G_1 = create_nx_graph(nodes_1,edges_1)
    an_adj, an_st, top_id, g_met_dict  = create_analysis(G_1,nodes_1)
    #l_data, l_dict = line_plot_data(an_adj)
    edges_1 = edge_styling(edges_1, top_id)
    #bar_1 = create_barchart(an_adj,'cent_value','cent_lab','cent_col')
    #bar_2 = create_barchart(an_adj,'rank_value','rank_lab','rank_col')
    #bar_3 = create_barchart(an_adj,'deg_value','deg_lab','deg_col')
    #bar_4 = create_barchart(an_adj,'betw_value','betw_lab','betw_col')
    #bar_dict = {1: bar_1, 2: bar_2, 3: bar_3, 4: bar_4}
    
    nodes_list = list()
    for i in range(len(nodes_1)):
        c_node = {
                "data": {"id": nodes_1.iloc[i,0], 
                          "e_cent_lab": an_adj.iloc[i,7],
                          "e_cent": an_adj.iloc[i,0], 
                          "e_col": an_adj.iloc[i,6],
                          "rank_lab": an_adj.iloc[i,9],
                          "rank": an_adj.iloc[i,1], 
                          "rank_col": an_adj.iloc[i,8],
                          "deg_lab": an_adj.iloc[i,11],
                          "deg": an_adj.iloc[i,2], 
                          "deg_col": an_adj.iloc[i,10],
                          "betw_lab": an_adj.iloc[i,13],
                          "betw": an_adj.iloc[i,3], 
                          "betw_col": an_adj.iloc[i,12]}
                
            }
        nodes_list.append(c_node)
    
    edges_list = list()
    for j in range(len(edges_1)):
        c_edge = {
                "data": {"source": edges_1.iloc[j,0], 
                          "target": edges_1.iloc[j,1],
                          "weight": edges_1.iloc[j,4], 
                          "color_cent": edges_1.iloc[j,7],
                          "color_rank": edges_1.iloc[j,8],
                          "color_deg": edges_1.iloc[j,9],
                          "color_betw": edges_1.iloc[j,10]}
            }
        edges_list.append(c_edge)
    
    elements = nodes_list + edges_list
    
    #return bar_dict, elements, g_met_dict, l_data, l_dict, an_adj, top_id
    return elements, g_met_dict, an_adj, top_id

def split_data_time(df_cat, start, end, time_step):
    #subset for greater than start
    #subset for less than end
    df_cat_sub = df_cat[(df_cat.DischargeDate <= end) & (df_cat.ReferralDate >= start)]
    
    #create list of time steps working backwards from end
    total_time = (pd.to_datetime(end) - pd.to_datetime(start))
    total_time_days = total_time.days
    periods = math.ceil(total_time_days / time_step)
    date_list = list()
    date_list.append(pd.to_datetime(end))
    time_step_delta = dt.timedelta(time_step)
    for i in range(periods-1):
        a = date_list[i] - time_step_delta
        date_list.append(a)
        
    #subset for less than and greater than through the list of timesteps
    #add time_step subsets to dictionary
    step_dict = dict()
    for j in range(len(date_list)-1):
        k = len(date_list) - j
        df_temp = df_cat_sub[(df_cat_sub.ReferralDate < date_list[j+1]) | ((df_cat_sub.ReferralDate >= date_list[j+1]) & (df_cat_sub.ReferralDate < date_list[j]))]
        step_dict[k] = df_temp[(df_temp.DischargeDate >= date_list[j]) | ((df_temp.DischargeDate >= date_list[j+1]) & (df_temp.DischargeDate < date_list[j]))]
        
    return step_dict

@app.callback([Output('no-team-comp', 'options'),
               Output('no-att-comp', 'options'),
               Output('no-team-spc', 'options'),
               Output('no-met-spc', 'options')],
              [Input('hidden-div', 'children')])
def node_comp_options(data):
    if data != None:
        clean_dict = json.loads(data)
        df_cat = preprocessing(clean_dict)
        servMove, singles = create_adjacency_matrix(df_cat)
        edgesdf = create_edge_list(servMove)
        nodesdf = create_node_list(df_cat)
        elements, g_met_dict, an_adj, top_id = net_analysis(nodesdf, edgesdf)
        
        team_list = list()
        for i in range(len(an_adj)):
            lab = an_adj.iloc[i,5]
            val = an_adj.iloc[i,4]
            team_list.append({'label': lab, 'value': val})
        
        metric_list = list()
        cols = list(an_adj.columns)
        cols = cols[0:4]
        for i in range(len(cols)):
            lab = str(cols[i])
            val = int(i)
            metric_list.append({'label': lab, 'value': lab})
            
        return team_list, metric_list, team_list, metric_list

@app.callback(Output('node-spc', 'figure'),
              [Input('no-team-spc', 'value'),
               Input('no-met-spc', 'value'),
               Input('spc-range', 'start_date'),
               Input('spc-range', 'end_date'),
               Input('spc-slice', 'value'),
               Input('hidden-div', 'children')])
def create_spc_chart(team, metric, start, end, time_step, data):
    if data != None and team != None and metric != None and start !=None and end != None and time_step != None:
        clean_dict = json.loads(data)
        df_cat = preprocessing(clean_dict)
        time_step = int(time_step)
        step_dict = split_data_time(df_cat, start, end, time_step)
        
        process_mean = []
        process_std = []
        
        for i in step_dict.keys():
            temp_df = step_dict[i]
            process_mean.append(temp_df['LoSdays'].mean())
            process_std.append(temp_df['LoSdays'].std())
        
        los_mean = stats.mean(process_mean)
        los_std = stats.mean(process_std)
        
        ucl = los_mean + 3 * los_std
        lcl = los_mean - 3 * los_std
        
        two_sd_above = los_mean + 2 * los_std
        two_sd_below = los_mean - 2 * los_std
        
        one_sd_above = los_mean + 1 * los_std
        one_sd_below = los_mean - 1 * los_std
        
        y_upper = ucl+10
        y_lower = lcl-10
        
        x_step = list(range(1,len(process_mean)+1))
        x_min = 0.5
        x_max = max(x_step)+0.5
        
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=x_step, y=process_mean, mode='lines+markers', name="Data", showlegend=True))
        # fig.add_trace(go.Scatter(x=[x_min,x_max], y=[los_mean,los_mean], mode='lines', name="Centre", showlegend=True))
        # fig.add_trace(go.Scatter(x=[x_min,x_max,None,x_min,x_max],y=[lcl,lcl,None,ucl,ucl], mode='lines', name="LCL/UCL", showlegend=True))
        # fig.add_trace(go.Scatter(x=[x_min,x_max,None,x_min,x_max],y=[two_sd_above,two_sd_above,None,two_sd_below,two_sd_below], mode='lines',name='2 \u03C3', showlegend=True))
        # fig.add_trace(go.Scatter(x=[x_min,x_max,None,x_min,x_max],y=[one_sd_above,one_sd_above,None,one_sd_below,one_sd_below], mode='lines',name='1 \u03C3', showlegend=True))
        # fig.update_layout(xaxis_title="Time step", yaxis_title="Deviation")
        
        actual_data = {
            "line": {
                "color": "rgb(0,116,217)", 
                "width": 2
                },
            "mode": "lines+markers",
            "name": "Data", 
            "type": "scatter",
            "x": x_step,
            "y": process_mean,
            "xaxis": "x", 
            "yaxis": "y", 
            "marker": {
                "size": 8, 
                "color": "rgb(0,116,217)", 
                "symbol": "circle"
                }, 
            "error_x": {"copy_ystyle": True}, 
            "error_y": {
                "color": "rgb(0,116,217)", 
                "width": 1, 
                "thickness": 1
                }, 
            "visible": True, 
            "showlegend": True
            }
        
        centre = {
            "line": {
                "color": "rgb(133,20,75)", 
                "width": 2
                },
            "mode": "lines", 
            "name": "Centre", 
            "type": "scatter",
            "x": [x_min,x_max],
            "y": [los_mean, los_mean],
            "xaxis": "x", 
            "yaxis": "y", 
            "marker": {
                "size": 8, 
                "color": "rgb(133,20,75)", 
                "symbol": "circle"
                }, 
            "error_x": {"copy_ystyle": True}, 
            "error_y": {
                "color": "rgb(133,20,75)", 
                "width": 1, 
                "thickness": 1
                }, 
            "visible": True, 
            "showlegend": True
            }
        
        limits = {
            "line": {
                "color": "rgb(235,88,52)", 
                "width": 2
                }, 
            "mode": "lines", 
            "name": "LCL/UCL", 
            "type": "scatter", 
            "x": [x_min, x_max, None, x_min, x_max], 
            "y": [lcl, lcl, None, ucl, ucl], 
            "xaxis": "x", 
            "yaxis": "y", 
            "marker": {
                "size": 8, 
                "color": "rgb(235,88,52)", 
                "symbol": "circle"
                }, 
            "error_x": {"copy_ystyle": True}, 
            "error_y": {
                "color": "rgb(235,88,52)", 
                "width": 1, 
                "thickness": 1
                }, 
            "visible": True, 
            "showlegend": True
            }
        
        two_sd = {
            "line": {
                "color": "rgb(235,137,52)", 
                "width": 2,
                "dash": "dash"
                }, 
            "mode": "lines", 
            "name": "2 \u03C3", 
            "type": "scatter", 
            "x": [x_min, x_max, None, x_min, x_max], 
            "y": [two_sd_below, two_sd_below, None, two_sd_above, two_sd_above], 
            "xaxis": "x", 
            "yaxis": "y", 
            "marker": {
                "size": 8, 
                "color": "rgb(235,137,52)", 
                "symbol": "circle"
                }, 
            "error_x": {"copy_ystyle": True}, 
            "error_y": {
                "color": "rgb(235,137,52)", 
                "width": 1, 
                "thickness": 1
                }, 
            "visible": True, 
            "showlegend": True
            }
        
        one_sd = {
            "line": {
                "color": "rgb(235,195,52)", 
                "width": 2,
                "dash": "dash"
                }, 
            "mode": "lines", 
            "name": "1 \u03C3", 
            "type": "scatter", 
            "x": [x_min, x_max, None, x_min, x_max], 
            "y": [one_sd_below, one_sd_below, None, one_sd_above, one_sd_above], 
            "xaxis": "x", 
            "yaxis": "y", 
            "marker": {
                "size": 8, 
                "color": "rgb(235,195,52)", 
                "symbol": "circle"
                }, 
            "error_x": {"copy_ystyle": True}, 
            "error_y": {
                "color": "rgb(235,195,52)", 
                "width": 1, 
                "thickness": 1
                }, 
            "visible": True, 
            "showlegend": True
            }

        #data = go.Data([actual_data, centre, limits, two_sd, one_sd])
        layout = {
            "title": "SPC Control Chart", 
            #"width": 952, 
            "xaxis": {
                "side": "bottom", 
                "type": "linear", 
                "range": [x_min, x_max], 
                "ticks": "inside", 
                "title": "Time step", 
                "anchor": "y", 
                "domain": [0.13, 0.746044], 
                "mirror": "allticks", 
                "showgrid": False, 
                "showline": True, 
                "zeroline": False, 
                "autorange": False, 
                "linecolor": "rgb(34,34,34)", 
                "linewidth": 1
                }, 
            "yaxis": {
                "side": "left", 
                "type": "linear", 
                "range": [y_lower, y_upper], 
                "ticks": "inside", 
                "title": "Deviation", 
                "anchor": "x", 
                "domain": [0.11, 0.925], 
                "mirror": "allticks", 
                "showgrid": False, 
                "showline": True, 
                "zeroline": False, 
                "autorange": True, 
                "linecolor": "rgb(34,34,34)", 
                "linewidth": 1
                }, 
            #"height": 930, 
            "legend": {
                "x": 0.19141689373297002, 
                "y": 0.7684964200477327, 
                #"xref": "paper", 
                #"yref": "paper", 
                "bgcolor": "rgba(255, 255, 255, 0.5)", 
                "xanchor": "left", 
                "yanchor": "bottom"
                }, 
            "margin": {
                "b": 0, 
                "l": 0, 
                "r": 0, 
                "t": 60, 
                "pad": 0, 
                "autoexpand": True
                }, 
            "autosize": True, 
            "showlegend": True, 
            "annotations": [
                {
                    "x": 0.438022, 
                    "y": 0.935, 
                    "text": "", 
                    # "title_xref": "paper", 
                    # "title_yref": "paper", 
                    "align": "center", 
                    "xanchor": "center", 
                    "yanchor": "bottom", 
                    "showarrow": False
                    }
                ], 
            "plot_bgcolor": "white", 
            "paper_bgcolor": "white"
            }
            
        dict_of_fig = dict({
            "data": [actual_data, centre, limits, two_sd, one_sd],
            "layout": layout
            })
        fig = go.Figure(dict_of_fig)
        
        return fig
    else:
        return go.Figure()
        

@app.callback([Output('node-bar', 'figure'),
               Output('no-comp-sum', 'children')],
              [Input('no-team-comp', 'value'),
                Input('no-att-comp', 'value')],
              [State('hidden-div', 'children')])
def create_bar_chart(team, metric, data):
    if data != None and team != None and metric != None:
        clean_dict = json.loads(data)
        df_cat = preprocessing(clean_dict)
        servMove, singles = create_adjacency_matrix(df_cat)
        edgesdf = create_edge_list(servMove)
        nodesdf = create_node_list(df_cat)
        elements, g_met_dict, an_adj, top_id = net_analysis(nodesdf, edgesdf)
        
        team_met_dict = an_adj.to_dict('index')
        fig = go.Figure()
        met_list = list()
        for i in range(len(team)):
            df = team_met_dict[team[i]]
            fig.add_trace(go.Bar(x=[df['lab']],
                                  y=[df[metric]]))
            met_list.append(df[metric])
        fig.update_layout(xaxis_title = 'Team', yaxis_title=metric, showlegend=False)
        
        met_df = pd.DataFrame(met_list, columns=['metric'])
        num_services = len(met_df)
        min_val = met_df['metric'].min()
        max_val = met_df['metric'].max()
        rng = max_val - min_val
        mean_att = met_df['metric'].mean()
        met_arr = np.array(met_list)
        met_dif = np.diff(met_arr)
        mean_dif = np.mean(met_dif)
        
        summary_list = [html.Li('Number of services: ' + str(num_services)),
                        html.Li('Minimum value: ' + str(round(min_val,2))),
                        html.Li('Maximum value: ' + str(round(max_val,2))),
                        html.Li('Range: ' + str(round(rng,2))),
                        html.Li('Mean value: ' + str(round(mean_att,2))),
                        html.Li('Mean difference: ' + str(round(mean_dif,2)))
                        ]
        
        return fig, summary_list
    else:
        return dash.no_update

# @app.callback(Output('node-bar', 'figure'),
#               [Input('hidden-div', 'children')])
# def create_bar_chart(data):
#     if data != None:
#         clean_dict = json.loads(data)
#         df_cat = preprocessing(clean_dict)
#         servMove, singles = create_adjacency_matrix(df_cat)
#         edgesdf = create_edge_list(servMove)
#         nodesdf = create_node_list(df_cat)
#         elements, g_met_dict, an_adj, top_id = net_analysis(nodesdf, edgesdf)
        
#         # team_met_dict = dict()
#         # for i in range(len())
#         team_met_dict = an_adj.to_dict('index')
#         #metric = str(metric)
#         fig = go.Figure()
#         df = team_met_dict[2]
#         # if df:
#         #     out = [html.Li('Data available')]
#         # else:
#         #     out = [html.Li('Empty')]
        
#         fig.add_trace(go.Bar(x=[df['lab']],
#                               y=[df['cent_value']]))
#         # fig.update_layout(xaxis_title = 'Team', yaxis_title='cent_value', showlegend=False)
#         # for i in range(len(team)):
#         #     df = team_met_dict[team[i]]
#         #     fig.add_trace(go.Bar(x=[df['lab']],
#         #                          y=[df[metric]]))
#         # fig.update_layout(xaxis_title = 'Team', yaxis_title=metric, showlegend=False)
        
#         return fig
#     else:
#         return dash.no_update
    
@app.callback(Output('static-net', 'elements'),
              [Input('hidden-div', 'children')])
def create_static_network(data):
    if data != None:
        clean_dict = json.loads(data)
        df_cat = preprocessing(clean_dict)
        servMove, singles = create_adjacency_matrix(df_cat)
        edgesdf = create_edge_list(servMove)
        nodesdf = create_node_list(df_cat)
        elements, g_met_dict, an_adj, top_id = net_analysis(nodesdf, edgesdf)
        
        return elements
    else:
        return list()
    

@app.callback([Output('net-sum','children')],
              [Input('hidden-div', 'children')])
def create_summaries(data):
    if data != None:
        clean_dict = json.loads(data)
        df_cat = preprocessing(clean_dict)
        servMove, singles = create_adjacency_matrix(df_cat)
        edgesdf = create_edge_list(servMove)
        nodesdf = create_node_list(df_cat)
        elements, g_met_dict, an_adj, top_id = net_analysis(nodesdf, edgesdf)
        net_sum_info_list = [html.Li('Number of nodes: ' + str(g_met_dict['num_chars'])),
                              html.Li('Number of edges: ' + str(g_met_dict['num_inter'])),
                              html.Li('Graph density: ' + str(round(g_met_dict['density'], 2)))]
        
        return net_sum_info_list
    else:
        return [html.Li('Awaiting data')]
    
@app.callback(Output('hidden-div', 'children'),
              [Input('test-button', 'n_clicks')],
              [State('store-two-data-upload', 'data')])
def hide_data_on_page(clicks,data):
    if clicks != 0:
        clean_dict = data
        # df_cat = preprocessing(clean_dict)
        # servMove, singles = create_adjacency_matrix(df_cat)
        # edgesdf = create_edge_list(servMove)
        # nodesdf = create_node_list(df_cat)
        # elements, g_met_dict, an_adj, top_id = net_analysis(nodesdf, edgesdf)
        # net_sum_info_list = [html.Li('Number of nodes: ' + str(g_met_dict['num_chars'])),
        #                       html.Li('Number of edges: ' + str(g_met_dict['num_inter'])),
        #                       html.Li('Graph density: ' + str(round(g_met_dict['density'], 2)))]
        
        return clean_dict

# @app.callback([Output('net-sum','children'),
#                Output('spc-sum', 'children')],
#               [Input('test-button', 'n_clicks')])
# def multi_output_test(n):
#     if n != 0:
#         return html.Li('Worked 1'), html.Li('Worked 2')

# @app.callback(Output('static-net', 'elements'),
#               [Input('store-two-data-upload', 'modified_timestamp')],
#               [State('store-two-data-upload', 'data')])
# @app.callback([Output('edges-store', 'data'),
#                Output('nodes-store', 'data'),
#                Output('analysis-df-store', 'data')],
#               [Input('store-two-data-upload', 'modified_timestamp')],
#               [State('store-two-data-upload', 'data')])
# def create_static_network(ts,data):
#     clean_dict = json.loads(data)
#     #clean_dict = [{k:v} for k, v in jdata.items()]
#     df_cat = preprocessing(clean_dict)
#     servMove, singles = create_adjacency_matrix(df_cat)
#     edgesdf = create_edge_list(servMove)
#     nodesdf = create_node_list(df_cat)
#     elements, g_met_dict, an_adj, top_id = net_analysis(nodesdf, edgesdf)
#     # net_sum_info_list = [html.Li('Number of nodes: ' + str(g_met_dict['num_chars'])),
#     #                      html.Li('Number of edges: ' + str(g_met_dict['num_inter'])),
#     #                      html.Li('Graph density: ' + str(round(g_met_dict['density'], 2)))]
#     j_edges = json.dumps(edgesdf)
#     j_nodes = json.dumps(nodesdf)
#     j_an_adj = json.dumps(an_adj)
#     j_g_met_dict = json.dumps(g_met_dict)
        
#     return j_edges, j_nodes, j_an_adj, j_g_met_dict

# @app.callback(Output('static-net', 'elements'),
#               [Input('edges-store', 'data'),
#                Input('nodes-store', 'data')])
# def visualise_network(edges, nodes):
#     nodes = json.loads(nodes)
#     edges = json.loads(edges)
#     elements, g_met_dict, an_adj, top_id = net_analysis(nodes, edges)
#     return elements

# @app.callback(Output('net-sum', 'children'),
#               [Input('g-met-store', 'data')])
# def graph_metrics(data):
#     g_met_dict = json.loads(data)
#     net_sum_info_list = [html.Li('Number of nodes: ' + str(g_met_dict['num_chars'])),
#                           html.Li('Number of edges: ' + str(g_met_dict['num_inter'])),
#                           html.Li('Graph density: ' + str(round(g_met_dict['density'], 2)))]
#     return net_sum_info_list

# @app.callback([Output('net-sum', 'children'),
#                Output('spc-sum', 'children')],
#               [Input('hidden-div', 'children')])
# def test_func(data):
#     data = data[0]
#     # clean_dict = json.loads(data)
#     # df_cat = preprocessing(clean_dict)
#     # servMove, singles = create_adjacency_matrix(df_cat)
#     # edgesdf = create_edge_list(servMove)
#     # nodesdf = create_node_list(df_cat)
#     # elements, g_met_dict, an_adj, top_id = net_analysis(nodesdf, edgesdf)
#     g_met_dict = json.loads(data)
#     # net_sum_info_list = [html.Li('Number of nodes: ' + str(g_met_dict['num_chars'])),
#     #                       html.Li('Number of edges: ' + str(g_met_dict['num_inter'])),
#     #                       html.Li('Graph density: ' + str(round(g_met_dict['density'], 2)))]
#     # spc_sum_info_list = [html.Li('Number of nodes: ' + str(g_met_dict['num_chars'])),
#     #                       html.Li('Number of edges: ' + str(g_met_dict['num_inter'])),
#     #                       html.Li('Graph density: ' + str(round(g_met_dict['density'], 2)))]
    
#     #return net_sum_info_list, spc_sum_info_list
#     return html.Li(str(type(g_met_dict))), html.Li(str(type(g_met_dict)))

