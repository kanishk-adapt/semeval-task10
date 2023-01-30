#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

task_b_long_labels = [
    '1. threats, plans to harm and incitement',
    '2. derogation',
    '3. animosity',
    '4. prejudiced discussions',
]

task_c_long_labels = [
    '1.1 threats of harm',
    '1.2 incitement and encouragement of harm',
    '2.1 descriptive attacks',
    '2.2 aggressive and emotive attacks',
    '2.3 dehumanising attacks & overt sexual objectification',
    '3.1 casual use of gendered slurs, profanities, and insults',
    '3.2 immutable gender differences and gender stereotypes',
    '3.3 backhanded gendered compliments',
    '3.4 condescending explanations or unwelcome advice',
    '4.1 supporting mistreatment of individual women',
    '4.2 supporting systemic discrimination against women as a group',
]

task_c_short_labels = []
task_c_short2long = {}
for label in task_c_long_labels:
    short_label = label[:3]
    task_c_short_labels.append(short_label)
    task_c_short2long[short_label] = label

label2index = {}
for index, label in enumerate(task_c_short_labels):
    label2index[label] = index
