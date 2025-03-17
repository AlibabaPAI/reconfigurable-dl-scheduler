```python
from tabulate import tabulate
import numpy as np
```


```python
execution_plans_small = ["DP", "GC","ZeRO-DP+GA","ZeRO-Offload"]
execution_plans_large = ["TP+PP", "DP+TP+PP","ZeRO-DP+GA","ZeRO-Offload+GC"]
```


```python
def draw_table(model_prediction,execution_plans):
    validation_data = []
    for execution_plan in execution_plans:
        if execution_plan not in model_prediction:
            validation_data.append([execution_plan,
                                "/",
                                "/"])
        else:
            validation_data.append([execution_plan,
                                round(np.average(model_prediction[execution_plan])*100,3),
                                round(np.max(model_prediction[execution_plan])*100,3)])

    headers = ["execution_plan", "avg.", "max."]
    table = tabulate(validation_data, headers, tablefmt="grid")
    return table

```

## ViT Validation


```python
from model.validate_vit import fit_ViT, validate
vit_fittable_params = fit_ViT()
vit_prediction = validate(vit_fittable_params)
table=draw_table(vit_prediction,execution_plans_small)
print(table)
```

    +------------------+--------+--------+
    | execution_plan   |   avg. |   max. |
    +==================+========+========+
    | DP               |  3.631 |  6.828 |
    +------------------+--------+--------+
    | GC               |  2.592 |  6.186 |
    +------------------+--------+--------+
    | ZeRO-DP+GA       |  4.231 |  6.669 |
    +------------------+--------+--------+
    | ZeRO-Offload     |  3.002 |  8.315 |
    +------------------+--------+--------+


## RoBERTa Validation


```python
from model.validate_roberta import fit_RoBERTa, validate
roberta_fittable_params = fit_RoBERTa()
roberta_prediction = validate(roberta_fittable_params)
table=draw_table(roberta_prediction,execution_plans_small)
print(table)
```

    +------------------+--------+--------+
    | execution_plan   |   avg. |   max. |
    +==================+========+========+
    | DP               |  2.209 |  4.365 |
    +------------------+--------+--------+
    | GC               |  3.358 |  4.292 |
    +------------------+--------+--------+
    | ZeRO-DP+GA       |  3.593 |  6.712 |
    +------------------+--------+--------+
    | ZeRO-Offload     |  7.418 | 10.436 |
    +------------------+--------+--------+


## BERT Validation


```python
from model.validate_bert import fit_BERT, validate
bert_fittable_params = fit_BERT()
bert_prediction = validate(bert_fittable_params)
table=draw_table(bert_prediction,execution_plans_small)
print(table)
```

    +------------------+--------+--------+
    | execution_plan   |   avg. |   max. |
    +==================+========+========+
    | DP               |  5.268 |  8.32  |
    +------------------+--------+--------+
    | GC               |  4.9   |  7.27  |
    +------------------+--------+--------+
    | ZeRO-DP+GA       |  3.701 |  6.899 |
    +------------------+--------+--------+
    | ZeRO-Offload     |  6.372 |  8.618 |
    +------------------+--------+--------+


## T5 Validation


```python
from model.validation_t5 import fit_T5, validate
t5_fittable_params = fit_T5()
t5_prediction = validate(t5_fittable_params)
table=draw_table(t5_prediction,execution_plans_large)
print(table)
```

    +------------------+--------+--------+
    | execution_plan   |   avg. |   max. |
    +==================+========+========+
    | TP+PP            |  3.184 |  8.239 |
    +------------------+--------+--------+
    | DP+TP+PP         |  2.406 |  9.55  |
    +------------------+--------+--------+
    | ZeRO-DP+GA       |  6.711 |  9.554 |
    +------------------+--------+--------+
    | ZeRO-Offload+GC  |  4.368 |  6.335 |
    +------------------+--------+--------+


## GPT-2 Validation


```python
from model.validate_gpt import fit_GPT2, validate
gpt2_fittable_params = fit_GPT2()
gpt2_prediction = validate(gpt2_fittable_params)
table=draw_table(gpt2_prediction,execution_plans_large)
print(table)
```

    +------------------+--------+--------+
    | execution_plan   |   avg. |   max. |
    +==================+========+========+
    | TP+PP            |  2.39  |  3.081 |
    +------------------+--------+--------+
    | DP+TP+PP         |  2.8   |  4.151 |
    +------------------+--------+--------+
    | ZeRO-DP+GA       |  2.516 |  3.855 |
    +------------------+--------+--------+
    | ZeRO-Offload+GC  |  5.521 |  8.901 |
    +------------------+--------+--------+


## LLaMA-2-7B Validation


```python
from model.validate_llama7B import fit_LLaMA7B, validate
llama7b_fittable_params = fit_LLaMA7B()
llama7b_prediction = validate(llama7b_fittable_params)
table=draw_table(llama7b_prediction,execution_plans_large)
print(table)
```

    +------------------+--------+--------+
    | execution_plan   | avg.   | max.   |
    +==================+========+========+
    | TP+PP            | 1.898  | 2.898  |
    +------------------+--------+--------+
    | DP+TP+PP         | 4.699  | 9.453  |
    +------------------+--------+--------+
    | ZeRO-DP+GA       | /      | /      |
    +------------------+--------+--------+
    | ZeRO-Offload+GC  | 4.094  | 6.382  |
    +------------------+--------+--------+


## LLaMA-30B Validation


```python
from model.validate_llama30B import fit_LLaMA30B, validate
llama30b_fittable_params = fit_LLaMA30B()
llama30b_prediction = validate(llama30b_fittable_params)
table=draw_table(llama30b_prediction,execution_plans_large)
print(table)
```

    +------------------+--------+--------+
    | execution_plan   | avg.   | max.   |
    +==================+========+========+
    | TP+PP            | 4.291  | 8.524  |
    +------------------+--------+--------+
    | DP+TP+PP         | 6.149  | 9.69   |
    +------------------+--------+--------+
    | ZeRO-DP+GA       | /      | /      |
    +------------------+--------+--------+
    | ZeRO-Offload+GC  | /      | /      |
    +------------------+--------+--------+



```python

```
