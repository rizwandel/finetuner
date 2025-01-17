# Labeler

Labeler is a component in Finetuner. It contains a backend and a frontend UI. Given {term}`unlabeled data` and an {term}`embedding model` or {term}`general model`, Labeler asks human for labeling data, trains model, conducts [active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) and then asks better questions for labeling.

Algorithms such as few-shot learning, negative sampling, active learning are implemented in the Labeler.

Labeler can also be used together with Tailor.

## `fit` method

### Use without Tailor

```{figure} labeler-case1.svg
:align: center
:width: 70%
```

```{code-block} python
---
emphasize-lines: 6
---
import finetuner

finetuner.fit(
    embed_model,
    train_data=unlabeled_data,
    interactive=True
)
```

### Use with Tailor

```{figure} labeler-case2.svg
:align: center
:width: 70%
```

```{code-block} python
---
emphasize-lines: 6, 7
---
import finetuner

finetuner.fit(
    general_model,
    train_data=labeled_data,
    interactive=True,
    to_embedding_model=True,
    freeze=False,
)
```

## Run Labeler interactively

Once you run the code above, you may get the following output in the console:
```console
      executor0@29672[W]: Using Thread as runtime backend is not recommended for production purposes. It is just supposed to be used for easier debugging. Besides the performance considerations, it isspecially dangerous to mix `Executors` running in different types of `RuntimeBackends`.
      executor1@29672[W]: Using Thread as runtime backend is not recommended for production purposes. It is just supposed to be used for easier debugging. Besides the performance considerations, it isspecially dangerous to mix `Executors` running in different types of `RuntimeBackends`.
        gateway@29672[W]: Using Thread as runtime backend is not recommended for production purposes. It is just supposed to be used for easier debugging. Besides the performance considerations, it isspecially dangerous to mix `Executors` running in different types of `RuntimeBackends`.
        gateway@29672[W]: The runtime HTTPRuntime will not be able to handle termination signals.  RuntimeError('set_wakeup_fd only works in main thread')
           Flow@29672[I]:🎉 Flow is ready to use!
	🔗 Protocol: 		HTTP
	🏠 Local access:	0.0.0.0:61130
	🔒 Private network:	172.18.1.109:61130
	🌐 Public address:	94.135.231.132:61130
	💬 Swagger UI:		http://localhost:61130/docs
	📚 Redoc:		http://localhost:61130/redoc
UserWarning: ignored unknown argument: ['thread']. (raised from /Users/hanxiao/Documents/jina/jina/helper.py:685)
⠴ Working... ━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:00 estimating...            JINA@29672[I]:Finetuner is available at http://localhost:61130/finetuner
```

All `UserWarning`s can be ignored. After a few seconds, your browser will open the Labeler UI. If not (depending on your operating system/browser setup), you can find the URL in the terminal and then open it manually. For example,

```console
JINA@29672[I]:Finetuner is available at http://localhost:61130/finetuner
```

```{tip}
While the frontend may already show examples to label, you may observe a progress bar on the backend that keeps showing `Working...`. This is because it is still loading your complete input data into the Labeler. The Labeler is designed in an "async" way so that you can directly start labeling without waiting for all data to load. 
```

If everything is successful, you should observe the following UI:

````{tab} Image 
```{figure} labeler-img.png
:align: center
```

````
````{tab} Text 
```{figure} labeler-text.png
:align: center
```

````

## User interface

The user interface of the Labeler is divided into two parts: the control panel on the left and question panel on the right.

### Control panel

Control panel is on the left side of the UI. It collects some configs of the frontend and backend to adjust your labeling experience.

#### View

The view section collects the configs determining how frontend renders the question panel.


````{sidebar} View
```{figure} control-view.png
:align: center
```
````

- `Field`: represents the field of `Document` your question data come from.
  - `Tags Key`: when you select `Field` as `.tags`, this textbox will show up, asking you to further specify which `.tags` key your question data comes from.
- `Content Type`: you need to select the right content type to have the correct rendering on the the question data.
- `Questions/session`: The maximum number of labeling examples on the frontend.
- `TopK/Question`: The maximum number of results for each example on the frontend.
- `Start question`: The starting index of the question
- `Keep same question`: If set, then `Start question` and `Questions/session` are locked. You will always get the same questions for labeling. 

````{tip}
If your question panel looks like the image below, this means rendering is not setup correctly. You need to change `Field`, `Content Type` and `Tags Key` to correct the render setup.

```{figure} bad-config.png
:align: center
:width: 50%
```

````

```{tip}
You can use `Keep same question` to debug your model: by fixing the query and observing how the model behaves after learning from your new labels.
```

#### Progress

Progress section collects the statistics of the labeling procedure so far.

````{sidebar} Progress
```{figure} control-progress.png
:align: center
```
````



- `This session`: the number of to-be-labeled examples in this session.
- `Done`: the number of labeled examples.
- `Positve`: the number of labeled positive instances.
- `Negative`: the number of labeled negative instances.
- `Ignore`: the number of ignored instances.
- `Saved`: how many times the model has been saved.

Below the stats there is a progress bar, indicating the ratio of positive, negative and ignored instances so far.

Click `Save Model` button to tell the backend to store the model weights at any time.

#### Advanced

````{sidebar} Advanced
```{figure} control-advanced.png
:align: center
```
````

In the advanced section, you can find some configs that affect the training procedure of the Tuner. Specifically:

- `Positive Label`: the value of the label when an instance is considered as positively related to the question.
- `Negative Label`: the value of the label when an instance is considered as negatively related/unrelated to the question.
- `Epochs`: the number of training epochs every time a new example is labeled.
- `Match pool`: the size of the pool for computing nearest neighbours. Note that a larger pool means more diversity when proposing a labeling question; yet it's slower on every proposal. A smaller pool means faster question proposal, but you may not have very meaningful questions if all top-K answers are bad.
- `Model save path`: the file path for saving the model, used when you click "Save model" button.

### Question panel


Question panel shows a multi-choice question in a card. The user needs to select the most relevant answers from the list/grid and submit the results.

```{figure} labeler-question.gif
:align: center
:width: 50%
```

```{figure} labeler-question-text.gif
:align: center
:width: 50%
```

You can use a keyboard shortcut to select related answers. The selections are considered positive, whereas the remains are considered negative. Use `Invert` or hit `<i>` to invert the selection.


Click `Done` or hit `<space>` to submit the result.

Once a submission is completed, you will see the backend starts to train based on your submission. A spinner will show near the "Progress" section, indicating the backend is working. Afterwards, a new question is proposed based on the newly trained model.




