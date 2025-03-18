_base_ = '../default.py'
train = dict(
        max_epoch=60,
        optimizer=dict(
            type='Adam',
            lr=1e-3, 
            weight_decay=1e-5),
        scheduler=dict(
                 lr_decay=dict(
                      steps=[30])      
            )
)