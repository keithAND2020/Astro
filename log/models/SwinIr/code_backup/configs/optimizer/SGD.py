_base_ = '../default.py'
train = dict(
        optimizer=dict(
            type='SGD',
            lr=1e-3, 
            weight_decay=1e-5,
            momentum=0.9),
        
)