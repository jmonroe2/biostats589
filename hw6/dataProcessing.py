#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 12:34:51 2018

@author: jmonroe

This script exists to complete biostats homework 6. Question order reflects the 
pdf available at /Users/jmonroe/Documents/classes/biostats/hw6
"""
import numpy as np
import matplotlib.pyplot as plt
import re ## see https://regexone.com/lesson for good tutorial

def grouped(in_list, num):
    ''' Returns a list iterated [num] objects at a time'''
    iterable = iter(in_list)
    return zip(*[iterable]*num)    
##END grouped


def main():
    ## load data
    data_dir = "/Users/jmonroe/Documents/classes/biostats/hw6/"
    file_name = "demo.txt"
    with open(data_dir+file_name) as open_file:
        read_lines = open_file.readlines()
    
    ## parse data
    sequence_list = []
    for text_block in grouped(read_lines,4):
        seq = Sequence(*text_block)
        sequence_list.append(seq)

    ## solve problems
    q1_count_errors(sequence_list)
    #q3_identify_errors(sequence_list)
    q6_identify_errors(sequence_list)
##END main
  
    
def q1_count_errors(sequence_list):
    error_counts = []
    for seq in sequence_list:
        q = seq.quality_chars
        #print(seq.seq_num, len(q)-q.count('E'), sum(seq.quality_ints)  )
        q_sum = sum(seq.quality_ints)
        error_counts.append(q_sum)
        print(seq.seq_num, q_sum)
        
    plt.plot(np.arange(1,len(error_counts)+1), error_counts)
    plt.xlabel("Sequence Number")
    plt.ylabel("Total quality factor")
    plt.show()
##END q1_count_errors()
    
    
def q3_identify_errors(sequence_list):
    standard = sequence_list[0].sequence
    line_length = 90
    
    print("%standard")
    print(standard[:line_length])
    print("\phantom{loremip}"+standard[line_length:])
    print("%Comparision")
    for Seq in sequence_list[1:]:
        to_compare = Seq.sequence
        highlighted = ''
        new_line_flag=False
        for i,letter in enumerate(to_compare):
            if i >=line_length and new_line_flag==False:
                highlighted += '\n' + "\phantom{loremip}"  
                new_line_flag = True
            if letter == standard[i]:
                highlighted += letter
            else:
                highlighted += r'\color{red}' + letter + r'\color{black}'
        ##END loop through nucleotides
        print()
        print(highlighted)
    ##END loop through (10) sequences
##END q3_count_errors
    
    
def q6_identify_errors(sequence_list):
    standard = sequence_list[0].sequence
    
    for Seq in sequence_list[1:]:
        to_compare = Seq.sequence
        print("Sequence "+str(Seq.seq_num)+":")
        for i,letter in enumerate(to_compare):
            if letter == standard[i]:
                pass
            else:
                q_char = Seq.quality_chars[i]
                q_int = Seq.quality_ints[i]
                print("\t Error @ nucleotide {0}: quality '{1}' ({2})"
                      .format(i+1,q_char,q_int))
        ##END loop through nucleotides
    ##END loop through (10) sequences
##END q6_identify_errors
        
    
class Sequence():
    def __init__(self,header,sequence, comments, quality):
        self.seq_num = int(header[4:6])
        self.sequence = sequence[:-1] ## remove \n
        self.length = len(sequence)
        self.quality_chars = quality[:-1]
        self.quality_ints = np.array([ord(c) for c in self.quality_chars ])
        self.quality_ints -= ord('!') ## subtract starting value
        self.quality_ints += 1 ## start from 1.
        self.prob = 10**(self.quality_ints/-10)
    ##END __init__
    
    def toString(self,do_print=False):
        out_str =  ""
        out_str += "Sequence "+ str(self.seq_num) +"\n"
        out_str += self.sequence + "\n"
        out_str += "Quality: "+ self.quality_chars
        if (do_print): print(out_str)
    ##END toString
##END Sequence  
    
if __name__ == '__main__':
    main()