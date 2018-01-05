require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'
require 'image'

require 'BatchIterator'
require 'utils'
-- require 'hdf5'

local config = dofile('config_one_image.lua')
config = config.parse(arg)
cutorch.setDevice(config.gpuid)

cutorch.setHeapTracking(true)

-- local model = dofile('model_multi_task.lua')(config.do_normal, config.do_semantic, config.do_boundary, config.do_room)
local model = dofile(config.model)(config) 

parameters, gradParameters = model:getParameters()
model:cuda()
parameters, gradParameters = model:getParameters()
parameters:copy(torch.load(config.test_model))

-- dataset
local train_data = {}
local test_data  = loadData(config.test_file, config)
local batch_iterator = BatchIterator(config, train_data, test_data)
batch_iterator:setBatchSize(1)

local test_count = 0

--while batch_iterator.epoch==0 and test_count<=config.max_count do
	local batch = batch_iterator:nextBatch('test', config)
	local currName = batch_iterator:currentName('test')

	local inputs = batch.input
    inputs = inputs:contiguous():cuda()
    local outputs = model:forward(inputs)

    local ch, h, w = 0, 0, 0
    local normal_est, normal_mask, normal_gnd, f_normal, df_do_normal, normal_outputs = nil,nil,nil,nil,nil,nil

    normal_est = outputs
    ch, h, w = normal_est:size(2), normal_est:size(3), normal_est:size(4)
    normal_est = normal_est:permute(1, 3, 4, 2):contiguous()
    normal_est = normal_est:view(-1, ch)
    local normalize_layer = nn.Normalize(2):cuda()
    normal_outputs = normalize_layer:forward(normal_est)
	normal_outputs = normal_outputs:view(1, h, w, ch)
	normal_outputs = normal_outputs:permute(1, 4, 2, 3):contiguous()
	normal_outputs = normal_outputs:view( ch, h, w)
	normal_outputs = normal_outputs:float()

	image.save(config.result_path, normal_outputs:add(1):mul(0.5))
    test_count = test_count + 1
--end
return config.result_path



