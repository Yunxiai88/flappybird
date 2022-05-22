# Flappy Bird

This project follows the description of the Deep Q Learning algorithm. we implemented it in Python using PyGame. The agent(bird) can only perform 2 actions: do nothing or jump. The simplicity of this problem makes it perfect for implementing reinforcement learning

# Technology
* Python: 3.7
* PyGame
* tensorflow==2.6.2
* keras==2.6.0
* opencv
* libraries like pandas, numpy

## Configuration
* 1. create a virtual environment:
    ```
    conda create -n flying python=3.7
    ```

* 2. activate newly created environment:
    ```
    conda activate flying
    ```

* 3. In the virtual environment, Go to the project root folder and run below command to install packages:
    ```
    pip install -r requirements.txt  
    ```

     If any packages fail to install, try installing individually
     If any errors, try to do this one more time to avoid packages being missed out

## Application Running:
* Execute the command below in the project SystemCode folder to start application:

```python
python main.py
```
## References
* 1. Peter Buttaroni, (Sep 22, 2020). Flappy Bird With DQN. Retrieved from https://www.deepalgos.com/2020/09/22/https-arxiv-org-pdf-1312-5602-pdf/ 

* 2. Anthony Li, (May 20, 2021). Reinforcement Learning in Python with Flappy Bird. Retrieved from https://towardsdatascience.com/reinforcement-learning-in-python-with-flappy-bird-37eb01a4e786 

* 3. Neven Pičuljan, (Apr 02, 2022). Schooling Flappy Bird: A Reinforcement Learning Tutorial, Retrieved from https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial 

