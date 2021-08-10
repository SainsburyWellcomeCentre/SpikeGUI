# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:09:11 2014

@author: crousse
"""

class Note(object):
    def __init__(self, notesData):
        self.folder = notesData['vars']['F_Folder'][0]
        self.stim = notesData['vars']['F_Stim'][0] 
        self.startTime = notesData['vars']['F_Tbgn'][0]
        self.endTime = notesData['vars']['F_Tend'][0]
        self.userName = notesData['vars']['H_Name'][0]
        self.lab = notesData['vars']['H_Lab'][0]
        self.temperature = notesData['vars']['F_Temp'][0]
