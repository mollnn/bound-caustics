#pragma once

#include "torch/torch.h"
#include "torch/script.h"
#include "shape.h"

// See https://pytorch.org/cppdocs/frontend.html#end-to-end-example for a brief example
class Encoder
{
public:
    inline explicit Encoder(const std::string &filePath)
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

    // compute latent code for a data vector
    torch::Tensor encode(const std::vector<float> &data)
    {
        if (data.size() != 18)
        {
            std::cout << "encode(data) error: size != 18" << std::endl;
            exit(1);
        }
        torch::Tensor feats = torch::zeros({1, 1, 18}).requires_grad_(false);
        for (int i = 0; i < 18; i++)
        {
            feats[0][0][i] = data[i];
        }
        torch::Tensor codes = model.forward({feats}).toTensor();
        return codes[0][0].unsqueeze(0);
    }

    torch::Tensor encode(const std::vector<std::vector<float>> &data)
    {
        int n = data.size();
        torch::Tensor latentCode = torch::zeros({1, 128}).requires_grad_(false);
        for (int i = 0; i < n; i++)
        {
            latentCode += encode(data[i]);
        }
        return latentCode;
    }

    // compute the latent code for each node inside the BVH of the shape
    void encode(MyShape *shape, int batchSize = 1) // ! fzm: disable batch (zero bias)
    {
        if (shape == nullptr)
        {
            return;
        }

        int totalNumTris = (int)shape->m_triangles.size();
        torch::Tensor feats = torch::zeros({batchSize, 1, 18}).requires_grad_(false);
        std::vector<torch::Tensor> latentCodes(totalNumTris);

        int i = 0;
        while (i < totalNumTris)
        {
            batchSize = std::min(batchSize, totalNumTris - i);

            // fill the batch data
            for (int j = 0; j < batchSize; ++j)
            {
                auto &tri = shape->m_triangles[i + j];
                // three vertices
                for (int k = 0; k < 3; ++k)
                {
                    // three dimensions
                    for (int d = 0; d < 3; ++d)
                    {
                        feats[j][0][k * 3 + d] = shape->m_verts[tri.index[k]][d];
                        feats[j][0][k * 3 + d + 9] = shape->m_vertexNormal[tri.index[k]][d];
                    }
                }
                std::cout << "triangle id " << i+j << std::endl;
                for(int k=0;k<18;k++) {
                    std::cout << " " << feats[j][0][k] << std::endl;
                }
            }

            // encoding
            if (batchSize != feats.size(0))
            {
                feats = feats.slice(0, 0, batchSize);
                assert(feats.size(0) == batchSize);
            }
            torch::Tensor codes = model.forward({feats}).toTensor();
            for (int j = 0; j < batchSize; ++j)
            {
                latentCodes[i + j] = codes[j][0].unsqueeze(0);
            }

            i += batchSize;
        }

        putLatentCodes(shape->m_root, latentCodes);
    }

private:
    torch::jit::script::Module model;

    // put the latent code into their corresponding tree node
    static void putLatentCodes(TreeNode *tree, const std::vector<torch::Tensor> &latentCodes)
    {
        if (tree == nullptr)
        {
            return;
        }

        if (tree->triangleIndex != -1)
        {
            tree->latentCode = latentCodes[tree->triangleIndex];
            std::cout << "triangle_latent " << tree->triangleIndex <<" " << tree->latentCode[0][0].item() << std::endl;
            return;
        }

        for (auto node : tree->child)
        {
            putLatentCodes(node, latentCodes);
        }

        // use sum pooling as an example
        torch::Tensor treeCode = torch::zeros({1, 128}).requires_grad_(false);
        for (auto node : tree->child)
        {
            if (node != nullptr)
            {
                treeCode += node->latentCode;
            }
        }
        tree->latentCode = treeCode;
        std::cout << "treenode_latent " << tree->nodeIndex <<" " << treeCode[0][0].item() << std::endl;
    }
};
