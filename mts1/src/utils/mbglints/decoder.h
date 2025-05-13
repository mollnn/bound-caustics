#pragma once

#include "torch/torch.h"
#include "torch/script.h"

class Decoder
{
public:
    inline explicit Decoder(const std::string &filePath)
    {
        // load the model from given path
        try
        {
            model = torch::jit::load(filePath);
        }
        catch (const c10::Error &e)
        {
            std::cerr << e.backtrace() << std::endl;
            std::exit(-1);
        }
        // set to evaluation mode
        model.eval();
    }

    torch::Tensor decode(Vec3 xD, Vec3 xL, const std::vector<torch::Tensor> &data)
    {
        int n = data.size();
        std::vector<torch::Tensor> tensors(n + 1);
        float xDxL[6] = {xD[0], xD[1], xD[2], xL[0], xL[1], xL[2]};
        tensors[0] = torch::from_blob(xDxL, 6).unsqueeze(0);
        for (int i = 0; i < n; i++)
        {
            tensors[i + 1] = data[i];
        }
        torch::Tensor input = torch::concat(tensors, 1);
        return model.forward({input}).toTensor();
    }

    // TODO decode the whole interval path, check sadtest04_train.py for input features
    // predict the throughput of a interval path
    torch::Tensor decode(const IntervalPath &path)
    {
        int nodeCount = path.getLength();
        auto &litPos = path.getPatch(0)->center;
        auto &camPos = path.getPatch(nodeCount - 1)->center;

        std::vector<torch::Tensor> tensors(nodeCount - 1);

        // xD + xL
        float xDxL[6] = {camPos[0], camPos[1], camPos[2], litPos[0], litPos[1], litPos[2]};
        // std::cout << xDxL[0] << " " << xDxL[1] << " " << xDxL[2] << " " 
        //           << xDxL[3] << " " << xDxL[4] << " " << xDxL[5] << std::endl;
        tensors[0] = torch::from_blob(xDxL, 6).unsqueeze(0);

        // latent codes
        for (int i = 1; i < nodeCount - 1; ++i)
        {
            // ! small end is camera patch
            // tensors[i] = path.getPatch(i)->latentCode;
            tensors[nodeCount - 1 - i] = path.getPatch(i)->latentCode;
        }

        torch::Tensor input = torch::concat(tensors, 1);
        return model.forward({input}).toTensor();
    }

private:
    torch::jit::script::Module model;
};
