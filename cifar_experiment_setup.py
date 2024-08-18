# basically copied and amended the Ignite tutorial to set up the basic experiment

import json
from typing import List, Callable
import os
import shutil
from . import IgniteModel
from . import make_cnn
import torch

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import global_step_from_engine

class ExperimentManager():
    def __init__(self, base_path:str, model_factory: Callable, model_factory_label:str, description=None, ds_code=None, ) -> None:
        self.base_path=base_path
        self.experiments=[]
        self.description=description
        self.ds_code=ds_code
        self.model_factory=model_factory
        self.model_factory_label=model_factory_label

    def set_trainset(self, trainset):
        self.trainset=trainset
    def set_validset(self,validset):
        self.validset=validset
    def set_evalset(self, evalset):
        self.evalset(evalset)
    def set_train_hyperpms(self, lrate, max_epoch, es_patience, min_delta, cumulative_delta):
        self.train_hyperpms={
            "lrate":lrate,
            "max_epoch": max_epoch,
            "es_patience":es_patience,
            "min_delta":min_delta,
            "cumulative_delta":cumulative_delta
        }
    def run_training_random_ds(self,model_kwargs,model_label, trainset_size,  split_seed,device=None):
        trainset=self.trainset.split_random(trainset_size, split_seed)
        dirpath=os.path.join(self.base_path,model_label)

        try:
            shutil.rmtree(dirpath)
        except:
            pass
        os.mkdir(dirpath)

        model_kwargs.update({"lr":self.train_hyperpms["lrate"]})

        model=self.model_factory(**model_kwargs)
        model.lr=self.train_hyperpms["lrate"]

        logs=set_up_and_run_training(
            model=model,
            run_prefix=model_label,
            train_loader=self.trainset.to_dloader(batch_size=128, shuffle=True),
            val_loader=self.validset.to_dloader(batch_size=128, shuffle=True),
            max_epoch=self.train_hyperpms["max_epoch"],
            earlystoping_patience=self.train_hyperpms["es_patience"],
            min_delta=self.train_hyperpms["min_delta"],
            cumulative_delta=self.train_hyperpms["cumulative_delta"],
            checkpoint_path=dirpath,
            device=device
            )
        with open(os.path.join(dirpath,"logs.json"),"w") as f:
            json.dump(logs,f,ensure_ascii=False)
        with open(os.path.join(dirpath,"hyperparameters.json"),"w") as f:
            json.dump({"model_kwargs": model_kwargs,
                       "seed_train": split_seed,}
                      ,f, ensure_ascii=False)
            
        self.experiments.append(model_label)

    def save_experiment(self):
        with open(os.path.join(self.base_path,"shared_hyperpm.json"),"w") as f:
            json.dump(self.train_hyperpms, f, ensure_ascii=False)
        with open(os.path.join(self.base_path,"suite_description.json"),"w") as f:
            json.dump({"description": self.description,
                       "dataset_code": self.ds_code,
                       "model_factory": self.model_factory_label,
                       "finished_experiments": self.experiments}, f, ensure_ascii=False)

class EvaluationManager():
    def __init__(self, suite_root_path: str, model_factory: Callable) -> None:

        """
        ensure that model factory is the same that was used to produce the experimental data
        """

        self.root_path=suite_root_path
        with open(os.path.join(suite_root_path, "shared_hyperpm.json"),"r") as f:
            self.shared_hyperpms=json.load(f)
        with open(os.path.join(suite_root_path, "suite_description.json"),"r") as f:
            self.description=json.load(f)

        self.model_factory=model_factory
        self.model_hyperparameters={}

    def load_models(self, model_label_list: List[str]):
        models={}
        for label in model_label_list:
            if label not in self.description["finished_experiments"]:
                raise ValueError("missing model")
            model_root_path=os.path.join(self.root_path, label)

            with open(os.path.join(model_root_path,"hyperparameters.json"),"r") as f:
                self.model_hyperparameters[label]=json.load(f)

            _model=self.model_factory(**self.model_hyperparameters[label]["model_kwargs"])
            
            dir_contents=os.listdir(model_root_path)
            _checkpoint=[f for f in dir_contents if f[-3:]==".pt"][0]
            checkpoint=os.path.join(model_root_path, _checkpoint)     
            checkpoint = torch.load(checkpoint)
            _model.load_state_dict(checkpoint)

            models[label]=_model
        
        return models


            

def cifar10_cnn_factory(c: int, lr:float = 1e-4):
    return IgniteModel(make_cnn,lr,torch.nn.CrossEntropyLoss(reduction="none"),c=c)

def predict_for_model_batch(models: List[IgniteModel], x:torch.Tensor, device="cpu"):
    x_dev=x.to(device=device)
    for m in models:
        m.to(device=device)
        m.eval()
    y=[m(x_dev) for m in models]

    return torch.stack(y, dim=0)


def set_up_and_run_training(model,run_prefix,train_loader,val_loader,
                            max_epoch,earlystoping_patience,min_delta,cumulative_delta,
                            log_interval=50,checkpoint_path='.',device=None,):

    if not device in ('cuda','cpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logs={"train_iter":list(),
          "train_epoch":list(),
          "valid_epoch":list(),}
    
    model=model.to(device)

    optimizer = model.configure_optimizers()
    criterion = model.pretty_loss

    main_trainer = create_supervised_trainer(model, optimizer, criterion, device)

    val_metrics = {"loss": Loss(criterion)}

    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")
        logs["train_iter"].append((engine.state.epoch,engine.state.iteration,engine.state.output))

    main_trainer.add_event_handler(Events.ITERATION_COMPLETED(every=log_interval),log_training_loss)

    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")
        logs["train_epoch"].append((trainer.state.epoch,metrics['loss']))

    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")
        logs["valid_epoch"].append((trainer.state.epoch,metrics['loss']))

    main_trainer.add_event_handler(Events.EPOCH_COMPLETED,log_training_results)
    main_trainer.add_event_handler(Events.EPOCH_COMPLETED,log_validation_results)


    def score_function(engine):
        return engine.state.metrics["loss"]*-1


    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        n_saved=1,
        filename_prefix=run_prefix,
        score_function=score_function,
        score_name="loss",
        require_empty=False,
        global_step_transform=global_step_from_engine(main_trainer),
    )

    es_handler = EarlyStopping(patience=earlystoping_patience, score_function=score_function, trainer=main_trainer,
                               min_delta=min_delta,cumulative_delta=cumulative_delta)
    
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    val_evaluator.add_event_handler(Events.COMPLETED, es_handler)
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    main_trainer.run(train_loader, max_epochs=max_epoch)

    return logs

def dummy_event_test():
    help(Events.COMPLETED)
    
        
        