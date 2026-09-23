import random
import numpy as np
from copy import deepcopy
from typing import Literal
import torch


class ClassIncrementalManager():
    def __init__(self, class_list: list[int], num_tasks: int, rand_seed: int = 0, shuffle=True, shuffle_level: Literal['class', 'task'] = 'class'):
        # assert len(class_list) % num_tasks == 0, f"{len(class_list)}, {num_tasks}"
        self.__rng = np.random.Generator(np.random.PCG64(rand_seed)) ## a random generator

        self.__task_class_list = np.array(class_list, dtype=np.int64) ## array: len=100, [0,1,2,...,99]

        if len(class_list) % num_tasks == 0:
            if shuffle:
                if shuffle_level == 'class':
                    self.__rng.shuffle(self.__task_class_list) ## shuffle class_list
                    self.__task_class_list = np.reshape(self.__task_class_list, [num_tasks, -1]) ## (10 tasks) * (10 classes)
                elif shuffle_level == 'task':
                    self.__task_class_list = np.reshape(self.__task_class_list, [num_tasks, -1])
                    self.__rng.shuffle(self.__task_class_list)
            else:
                self.__task_class_list = np.reshape(self.__task_class_list, [num_tasks, -1])

            for taskid in range(num_tasks):
                print('Task-id:{} || Training labels:{}'.
                    format(taskid+1, list(self.__task_class_list[taskid])))

            self.__all_classes = self.__task_class_list.flatten().tolist() ## shuffled class list, len=100
            self.__task_class_list = self.__task_class_list.tolist() ## len=10, len(self.__task_class_list[0])=10, 10*10 split

        else:
            self.__rng.shuffle(self.__task_class_list)
            self.__all_classes = self.__task_class_list
            num_classes_per_task = int(len(class_list) // num_tasks + 1)
            self.__class_list = []
            for taskid in range(num_tasks):
                if taskid == num_tasks - 1:
                    self.__class_list.append(self.__task_class_list[taskid * num_classes_per_task:])
                else:
                    self.__class_list.append(self.__task_class_list[taskid * num_classes_per_task : (taskid+1) * num_classes_per_task])
            self.__task_class_list = self.__class_list

        self.__current_taskid = -1
        self.__num_tasks = num_tasks ## split number=10

        self.__class_task_dict = torch.zeros(class_list.__len__(), dtype=torch.int64)
        for taskid in range(len(self.task_class_list)):
            for classid in self.task_class_list[taskid]:
                self.__class_task_dict[classid] = int(taskid)

        self.storage = {}

    @property
    def current_taskid(self) -> int:
        assert self.__current_taskid >= 0, "Not initialized"
        return self.__current_taskid

    @property
    def all_classes(self) -> list[int]:
        return self.__all_classes

    @property
    def task_class_list(self) -> list[list[int]]:
        return self.__task_class_list
    
    @property
    def class_task_dict(self) -> list[list[int]]:
        return self.__class_task_dict

    @property
    def num_tasks(self) -> int:
        return self.__num_tasks

    @property
    def num_classes_per_task(self) -> int:
        return len(self.__task_class_list[0])

    @property
    def current_task_classes(self) -> list[int]:
        return deepcopy(self.__task_class_list[self.current_taskid])

    @property
    def sofar_task_classes(self) -> list[list[int]]:
        extend = True
        classes = []
        for i in range(self.current_taskid + 1):
            if extend:
                classes.extend(self.__task_class_list[i])  ## current labels
            else:
                classes.append(self.__task_class_list[i])
        return deepcopy(classes)

    def get_classes(self, taskid: int) -> list[int]:
        return self.__task_class_list[taskid]

    def __iter__(self):
        return self

    def __next__(self):
        self.__current_taskid += 1
        if self.current_taskid >= len(self): ## len(self) = num_tasks
            self.__current_taskid = -1
            raise StopIteration()
        return self.current_taskid, self.current_task_classes

    def __len__(self) -> int:
        return self.num_tasks
