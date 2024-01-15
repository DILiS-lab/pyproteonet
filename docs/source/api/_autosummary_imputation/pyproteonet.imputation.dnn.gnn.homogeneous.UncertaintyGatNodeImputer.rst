pyproteonet.imputation.dnn.gnn.homogeneous.UncertaintyGatNodeImputer
====================================================================

.. currentmodule:: pyproteonet.imputation.dnn.gnn.homogeneous

.. autoclass:: UncertaintyGatNodeImputer
   :members:

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~UncertaintyGatNodeImputer.__init__
      ~UncertaintyGatNodeImputer.add_module
      ~UncertaintyGatNodeImputer.all_gather
      ~UncertaintyGatNodeImputer.apply
      ~UncertaintyGatNodeImputer.backward
      ~UncertaintyGatNodeImputer.bfloat16
      ~UncertaintyGatNodeImputer.buffers
      ~UncertaintyGatNodeImputer.calculate_loss
      ~UncertaintyGatNodeImputer.children
      ~UncertaintyGatNodeImputer.clip_gradients
      ~UncertaintyGatNodeImputer.compile
      ~UncertaintyGatNodeImputer.configure_callbacks
      ~UncertaintyGatNodeImputer.configure_gradient_clipping
      ~UncertaintyGatNodeImputer.configure_model
      ~UncertaintyGatNodeImputer.configure_optimizers
      ~UncertaintyGatNodeImputer.configure_sharded_model
      ~UncertaintyGatNodeImputer.cpu
      ~UncertaintyGatNodeImputer.cuda
      ~UncertaintyGatNodeImputer.double
      ~UncertaintyGatNodeImputer.eval
      ~UncertaintyGatNodeImputer.extra_repr
      ~UncertaintyGatNodeImputer.float
      ~UncertaintyGatNodeImputer.forward
      ~UncertaintyGatNodeImputer.freeze
      ~UncertaintyGatNodeImputer.get_buffer
      ~UncertaintyGatNodeImputer.get_extra_state
      ~UncertaintyGatNodeImputer.get_parameter
      ~UncertaintyGatNodeImputer.get_submodule
      ~UncertaintyGatNodeImputer.half
      ~UncertaintyGatNodeImputer.ipu
      ~UncertaintyGatNodeImputer.load_from_checkpoint
      ~UncertaintyGatNodeImputer.load_state_dict
      ~UncertaintyGatNodeImputer.log
      ~UncertaintyGatNodeImputer.log_dict
      ~UncertaintyGatNodeImputer.lr_scheduler_step
      ~UncertaintyGatNodeImputer.lr_schedulers
      ~UncertaintyGatNodeImputer.manual_backward
      ~UncertaintyGatNodeImputer.modules
      ~UncertaintyGatNodeImputer.named_buffers
      ~UncertaintyGatNodeImputer.named_children
      ~UncertaintyGatNodeImputer.named_modules
      ~UncertaintyGatNodeImputer.named_parameters
      ~UncertaintyGatNodeImputer.on_after_backward
      ~UncertaintyGatNodeImputer.on_after_batch_transfer
      ~UncertaintyGatNodeImputer.on_before_backward
      ~UncertaintyGatNodeImputer.on_before_batch_transfer
      ~UncertaintyGatNodeImputer.on_before_optimizer_step
      ~UncertaintyGatNodeImputer.on_before_zero_grad
      ~UncertaintyGatNodeImputer.on_fit_end
      ~UncertaintyGatNodeImputer.on_fit_start
      ~UncertaintyGatNodeImputer.on_load_checkpoint
      ~UncertaintyGatNodeImputer.on_predict_batch_end
      ~UncertaintyGatNodeImputer.on_predict_batch_start
      ~UncertaintyGatNodeImputer.on_predict_end
      ~UncertaintyGatNodeImputer.on_predict_epoch_end
      ~UncertaintyGatNodeImputer.on_predict_epoch_start
      ~UncertaintyGatNodeImputer.on_predict_model_eval
      ~UncertaintyGatNodeImputer.on_predict_start
      ~UncertaintyGatNodeImputer.on_save_checkpoint
      ~UncertaintyGatNodeImputer.on_test_batch_end
      ~UncertaintyGatNodeImputer.on_test_batch_start
      ~UncertaintyGatNodeImputer.on_test_end
      ~UncertaintyGatNodeImputer.on_test_epoch_end
      ~UncertaintyGatNodeImputer.on_test_epoch_start
      ~UncertaintyGatNodeImputer.on_test_model_eval
      ~UncertaintyGatNodeImputer.on_test_model_train
      ~UncertaintyGatNodeImputer.on_test_start
      ~UncertaintyGatNodeImputer.on_train_batch_end
      ~UncertaintyGatNodeImputer.on_train_batch_start
      ~UncertaintyGatNodeImputer.on_train_end
      ~UncertaintyGatNodeImputer.on_train_epoch_end
      ~UncertaintyGatNodeImputer.on_train_epoch_start
      ~UncertaintyGatNodeImputer.on_train_start
      ~UncertaintyGatNodeImputer.on_validation_batch_end
      ~UncertaintyGatNodeImputer.on_validation_batch_start
      ~UncertaintyGatNodeImputer.on_validation_end
      ~UncertaintyGatNodeImputer.on_validation_epoch_end
      ~UncertaintyGatNodeImputer.on_validation_epoch_start
      ~UncertaintyGatNodeImputer.on_validation_model_eval
      ~UncertaintyGatNodeImputer.on_validation_model_train
      ~UncertaintyGatNodeImputer.on_validation_model_zero_grad
      ~UncertaintyGatNodeImputer.on_validation_start
      ~UncertaintyGatNodeImputer.optimizer_step
      ~UncertaintyGatNodeImputer.optimizer_zero_grad
      ~UncertaintyGatNodeImputer.optimizers
      ~UncertaintyGatNodeImputer.parameters
      ~UncertaintyGatNodeImputer.predict_dataloader
      ~UncertaintyGatNodeImputer.predict_step
      ~UncertaintyGatNodeImputer.prepare_data
      ~UncertaintyGatNodeImputer.print
      ~UncertaintyGatNodeImputer.register_backward_hook
      ~UncertaintyGatNodeImputer.register_buffer
      ~UncertaintyGatNodeImputer.register_forward_hook
      ~UncertaintyGatNodeImputer.register_forward_pre_hook
      ~UncertaintyGatNodeImputer.register_full_backward_hook
      ~UncertaintyGatNodeImputer.register_full_backward_pre_hook
      ~UncertaintyGatNodeImputer.register_load_state_dict_post_hook
      ~UncertaintyGatNodeImputer.register_module
      ~UncertaintyGatNodeImputer.register_parameter
      ~UncertaintyGatNodeImputer.register_state_dict_pre_hook
      ~UncertaintyGatNodeImputer.requires_grad_
      ~UncertaintyGatNodeImputer.save_hyperparameters
      ~UncertaintyGatNodeImputer.set_extra_state
      ~UncertaintyGatNodeImputer.setup
      ~UncertaintyGatNodeImputer.share_memory
      ~UncertaintyGatNodeImputer.state_dict
      ~UncertaintyGatNodeImputer.teardown
      ~UncertaintyGatNodeImputer.test_dataloader
      ~UncertaintyGatNodeImputer.test_step
      ~UncertaintyGatNodeImputer.to
      ~UncertaintyGatNodeImputer.to_empty
      ~UncertaintyGatNodeImputer.to_onnx
      ~UncertaintyGatNodeImputer.to_torchscript
      ~UncertaintyGatNodeImputer.toggle_optimizer
      ~UncertaintyGatNodeImputer.train
      ~UncertaintyGatNodeImputer.train_dataloader
      ~UncertaintyGatNodeImputer.training_step
      ~UncertaintyGatNodeImputer.transfer_batch_to_device
      ~UncertaintyGatNodeImputer.type
      ~UncertaintyGatNodeImputer.unfreeze
      ~UncertaintyGatNodeImputer.untoggle_optimizer
      ~UncertaintyGatNodeImputer.val_dataloader
      ~UncertaintyGatNodeImputer.validation_step
      ~UncertaintyGatNodeImputer.xpu
      ~UncertaintyGatNodeImputer.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~UncertaintyGatNodeImputer.CHECKPOINT_HYPER_PARAMS_KEY
      ~UncertaintyGatNodeImputer.CHECKPOINT_HYPER_PARAMS_NAME
      ~UncertaintyGatNodeImputer.CHECKPOINT_HYPER_PARAMS_TYPE
      ~UncertaintyGatNodeImputer.T_destination
      ~UncertaintyGatNodeImputer.automatic_optimization
      ~UncertaintyGatNodeImputer.call_super_init
      ~UncertaintyGatNodeImputer.current_epoch
      ~UncertaintyGatNodeImputer.device
      ~UncertaintyGatNodeImputer.dtype
      ~UncertaintyGatNodeImputer.dump_patches
      ~UncertaintyGatNodeImputer.example_input_array
      ~UncertaintyGatNodeImputer.fabric
      ~UncertaintyGatNodeImputer.global_rank
      ~UncertaintyGatNodeImputer.global_step
      ~UncertaintyGatNodeImputer.hparams
      ~UncertaintyGatNodeImputer.hparams_initial
      ~UncertaintyGatNodeImputer.local_rank
      ~UncertaintyGatNodeImputer.logger
      ~UncertaintyGatNodeImputer.loggers
      ~UncertaintyGatNodeImputer.model
      ~UncertaintyGatNodeImputer.on_gpu
      ~UncertaintyGatNodeImputer.out_dim
      ~UncertaintyGatNodeImputer.trainer
      ~UncertaintyGatNodeImputer.training
   
   