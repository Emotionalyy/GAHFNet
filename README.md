# GAHFNet
## Start
train：

just run "mytrain_VGG_CLAHE.py"

1.Pseudo-label train:turn the "parser--train_path" to "Pseudo-label".

	Turn the "parser--is_semi" to "False", 

	Turn the "parse--is_pseudo" to "True".

2.Doctor-label train:turn the "parser--train_path" to "Doctor-label".

	Turn the "parser--is_semi" to "True", 

	Turn the "parse--is_pseudo" to "False".    

test:just run "mytest_VGG_CLAHE.py"

----------------------------------------------------------------

## About the use of data sets

train： just run "mytrainVGG" 

1.Pseudo-labeltrain:

	Turn the "parser--train_path" to "Pseudo-label".

	Turn the "parser--is_semi" to "False",

	Turn the "parse--is_pseudo" to "False".
2.Doctor-label train:

	Turn the "parser--train_path" to "Doctor-label".

	Turn the "parser--is_semi" to "True",

	Turn the "parse--is_pseudo" to "False".

test: just run "mytest_VGG.py"

Other models are similar

----------------------------------------------------------------

If you find our work helpful, please cite it in your paper

Welcome to exchange: yyyyaixue@163.com
