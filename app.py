#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues June 23 10:40:49 2020

@author: sean
"""

import dash

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server