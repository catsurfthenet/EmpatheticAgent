# EmpatheticAgent

## Download Packages
`pip install requirements.txt`

If encountered any issues please contact me at: cn222@ic.ac.uk

## Download PromCSE
See: https://github.com/YJiangcm/PromCSE

## Download BLEURT
See: https://github.com/google-research/bleurt

## Download Datasets and Empathy Classifier

## Training Code
It is recommended to run the supervised learning approach 
as it is more stable and performs better in general. 
### Supervised Learning Approach
#### With context
`python sl_ft_conv.py`

#### Without context
`python sl_ft_no_conv.py`

#### Without emotion special tokens
`python sl_ft_no_sp.py`

### Reinforcement Learning Approach
#### Empathetic Utterance Generation Task
`python rl_ppo_empathy.py`

#### Short Utterance Generation Task
`python rl_ppo_short.py`

## Evaluation/ Test Code
Move the test files out of the `/tests` directory 
to the main directory. Run 

`python test_model.py`

for the standard test, and run

`python test_model_bleurt.py`

for test with BLEURT metric. Note that the BLEURT test
takes much longer than the standard test.

