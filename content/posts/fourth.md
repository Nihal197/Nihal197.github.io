---
title: "How to EFFICIENTLY use Pandas for your work"
date: 2022-07-23T23:32:19+05:30
draft: true
ShowToc: true
math: true
tags: ["Data Wrangling", "Pandas"]
---

## Preface 

While working in the ever fast changing world of tech, first thing a begineer in Data science will come to face is data and it's specialized "requirements".
When I started, I had no idea how would one wrangle or play with the data. I was told to checkout [Pandas Cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf), but when I was going through cheatsheets and the data I had in hand, I realised that this could be the trap for begineers to easily fall in.

What I want you to know is, it's great to start with a cheatsheet because the documentation itself might become overwhelming. But please, once you are comfortable with the cheatsheet syntaxes, take some time off and read the documentation. It is beautifully designed and very beginner friendly with the tutorials. 

Let's say you want to start with Pandas, you know what a DataFrame is, what the most common syntaxes for it are, you can start reading documentation for DataFrame itself.
What this does is, it slowly builds a habit for you to read from documentation directly and not relying upon the video tutorials. I kid you not, when you'll come to the job you will have to catch up real fast and you might not be getting cheesy tutorials for each of the tools they are using. But, if you are comfortable with the documentation itself, you'll become unstoppable. 

There is next step to it, I personally do not recommend it but in case the need arises, you might have to look at the library itself to find what syntax does (won't be happing almost all the time).

## Pre-requisite 
There is none assumed except the reader's perseverance. 


## Overview 

Let's start with the big boy in python's data library Pandas. In this article I'll try to cover most of the stuff you'd have to know to ease your coding journey. From input to extraction, how to get comfortable and play with data, HOW TO VIEW THEM, what will be the syntaxes you must know, plotting  and in the end how to export it. That basically will add another skill in your skillset. And hopefully less hours in the job figuring out how to do instead of doing it asap and playing games rest of the day.

## Data Structures 

First we import the library, pd is natural convention. Keep in mind one thing, whatever you do with your data, the data alignment will be intrinsic. So, you don't have to worry about it getting misplaced during any operation unless you explicitly wants it to.

Basically two kinds of structures we will go through -
-item Series 
-item DataFrame

### Series 

You are already familiar with Vectors, series is basiaclly 1-D vector (but it comes with labels). For example - 
```Python
import pandas as pd #pd is natural convention 

s = pd.Series(data, index = index)
```

`data` can be any of the three things- It can be a dict, or an ndarry (numpy array), or simply a value (10). `index` is axis labels. Example will make it clear.

```
In [1]: d = {"b": 1, "a": 0, "c": 2} #examples taken from the documentation itself, so feel free to checkout more 

In [2]: pd.Series(d)

Out[2]: 
b    1
a    0
c    2
dtype: int64
```


 

## Import 

Pandas can import different kinds of data, let's start with the common ones. 
