# basically copied and amended the Ignite tutorial to set up the basic experiment

import json
import os
import pickle
import shutil
import torch

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import global_step_from_engine

class ExperimentManager():
    def __init__(self, base_path:str, description=None, ds_code=None) -> None:
        self.base_path=base_path
        self.experiments=[]
        self.description=description
        self.ds_code=ds_code

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
    def run_training_random_ds(self,model,model_label, trainset_size,  split_seed,device=None):
        trainset=self.trainset.split_random(trainset_size, split_seed)
        dirpath=os.path.join(self.base_path,model_label)

        try:
            shutil.rmtree(dirpath)
        except:
            pass
        os.mkdir(dirpath)


        logs=set_up_and_run_training(
            model=model,
            run_prefix=model_label,
            train_loader=trainset.to_dloader(batch_size=128, shuffle=True),
            val_loader=self.set_validset.to_dloader(batch_size=128, shuffle=True),
            max_epoch=self.train_hyperpms["max_epoch"],
            earlystoping_patience=self.train_hyperpms["es_patience"],
            min_delta=self.train_hyperpms["min_delta"],
            cumulative_delta=self.train_hyperpms["cumulative_delta"],
            checkpoint_path=dirpath,
            device=device
            )
        with open(os.path.join(dirpath,"logs.json"),"w") as f:
            json.dump(logs,f)
        with open(os.path.join(dirpath,"hyperparameters.json"),"w") as f:
            json.dump({"c":model.c,
                          "lr":model.lr,
                        "seed_train": split_seed,}.update(self.train_hyperpms)
                      ,f)
            
        self.experiments.append(model_label)

    def save_experiment(self):
        with open(os.path.join(self.base_path,"shared_hyperpm.json"),"w") as f:
            json.dump(self.train_hyperpms, f)
        with open(os.path.join(self.base_path,"suite_description.json"),"w") as f:
            json.dump({"description": self.description,
                       "dataset_code": self.ds_code,
                       "finished_experiments": self.experiments}, f)



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
    
        
        