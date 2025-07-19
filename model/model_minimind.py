# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind配置类：像DNA一样定义了模型的基因蓝图
    
    每个参数就像生物基因中的碱基对，决定了这个AI大脑的：
    - 神经元数量（hidden_size）
    - 记忆容量（max_position_embeddings）
    - 语言能力（vocab_size）
    - 专家系统（MOE相关参数）
    """
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,  # 🎯 神经元"死亡疫苗"：0.0表示不打疫苗，让神经元100%存活
            bos_token_id: int = 1,  # 🏁 起跑线标记：1表示"开始说话"的哨声
            eos_token_id: int = 2,  # 🏁 终点线标记：2表示"结束说话"的哨声
            hidden_act: str = 'silu',  # 🧪 神经元"激活酶"：SiLU像温和的消化酶，避免信息"消化不良"
            hidden_size: int = 512,  # 🧠 每个神经元的"树突数量"：512根天线接收信号
            intermediate_size: int = None,  # 🏭 前馈网络的"胃容量"：None时自动计算为1365
            max_position_embeddings: int = 32768,  # 📏 最长"记忆长度"：能记住32768个词的位置
            num_attention_heads: int = 8,  # 👁️ 8只"注意力眼睛"：同时观察8个不同角度
            num_hidden_layers: int = 8,  # 🏢 8层"大脑皮层"：每层处理不同抽象级别
            num_key_value_heads: int = 2,  # 🔑 2组"钥匙保管员"：减少内存占用的优化技巧
            vocab_size: int = 6400,  # 📚 掌握的"词汇量"：6400个词的中文词典
            rms_norm_eps: float = 1e-05,  # ⚖️ 标准化"精度调节器"：防止除零的极小值
            rope_theta: int = 1000000.0,  # 🌪️ 旋转编码的"频率基数"：100万像无线电的基准频率
            flash_attn: bool = True,  # ⚡ 是否启用"闪电注意力"：True表示用GPU加速计算
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,  # 🏥 是否启用"专家会诊系统"：False表示所有问题都用同一个大脑
            num_experts_per_tok: int = 2,  # 👨‍⚕️ 每个词选2个专家：像看病时挂2个科室的号
            n_routed_experts: int = 4,  # 🏥 总共有4个专家：内科、外科、神经科、心理科
            n_shared_experts: int = 1,  # 👨‍⚕️ 1个全科医生：处理所有基础问题
            scoring_func: str = 'softmax',  # 📊 专家评分函数：softmax像"投票系统"
            aux_loss_alpha: float = 0.1,  # ⚖️ 负载均衡"调节器"：防止某个专家太忙
            seq_aux: bool = True,  # 📏 是否在序列级别计算辅助损失：True表示按句子统计
            norm_topk_prob: bool = True,  # 📊 是否标准化top-k概率：True确保权重总和为1
            **kwargs
    ):
        super().__init__(**kwargs)
        # 💾 将所有基因参数保存到实例变量，像DNA转录到RNA
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    🎯 神经元"血压调节器"：确保信号强度在安全范围内
    
    工作原理像智能稳压器：
    1. 计算输入信号的"平均能量"（平方均值）
    2. 生成"调节系数"（平方根倒数）
    3. 用系数校准信号强度，防止"电压过高"或"过低"
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 🛡️ 防除零保护：像电路中的保险丝
        self.weight = nn.Parameter(torch.ones(dim))  # ⚖️ 可学习权重：像可调电阻

    def _norm(self, x):
        """
        🧮 核心计算：像计算"信号强度调节器"
        
        详细步骤：
        1. x.pow(2) → 每个元素平方（像计算能量）
        2. .mean(-1, keepdim=True) → 计算最后一个维度的平均值（像平均能量）
        3. + self.eps → 加极小值防除零（像保险丝）
        4. torch.rsqrt → 平方根倒数（像调节系数）
        5. x * ... → 原信号乘以调节系数（像校准电压）
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        🔄 前向传播：像"信号标准化流水线"
        
        1. 先把输入转换成float32（高精度计算）
        2. 应用标准化公式
        3. 转回原始数据类型（节省内存）
        4. 乘以可学习权重（个性化调节）
        """
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    🌪️ 预计算旋转位置编码：像给每个位置生成"GPS坐标"
    
    参数说明：
    - dim: 每个注意力头的维度（通常是64）
    - end: 最大序列长度（32768像支持32K上下文）
    - theta: 频率基数（100万像无线电的基准频率）
    
    计算过程像"频率生成器"：
    1. torch.arange(0, dim, 2) → 生成[0,2,4,...,62]（步长2采集）
    2. / dim → 归一化到[0,1]范围
    3. theta ** (...) → 生成不同频率（像不同波长的无线电）
    4. torch.outer → 为每个位置计算频率组合
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # 📏 位置索引：[0,1,2,...,32767]
    freqs = torch.outer(t, freqs).float()  # 🎯 每个位置×每个频率 = 位置频率矩阵
    
    # 🎭 生成正弦/余弦表：像32768个预设的"位置指纹"
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # 余弦坐标
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # 正弦坐标
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    🎠 应用旋转位置编码：像给词向量做"旋转木马"
    
    核心思想：每个位置的向量被旋转特定角度，但保持相对距离
    
    rotate_half函数：像把向量对折后旋转180度
    - 输入：[a,b,c,d] → 输出：[-c,-d,a,b]
    """
    def rotate_half(x):
        """🔄 向量旋转180度：后半部分取负并交换位置"""
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 🎯 应用旋转：像给每个查询和键加上"位置旋转"
    # 公式：(q * cos) + (rotate_half(q) * sin) 像复数旋转的实数实现
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    🔁 重复键/值：像复印机一样复制信息
    
    技术背景：当num_key_value_heads < num_attention_heads时，
    需要复制KV来匹配Q的头数
    
    参数：
    - x: 形状[batch_size, seq_len, num_kv_heads, head_dim]
    - n_rep: 每个KV头需要重复的次数
    
    返回：
    - 形状[batch_size, seq_len, num_kv_heads*n_rep, head_dim]
    """
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x  # 🎯 无需重复，直接返回原样
    return (
        x[:, :, :, None, :]  # 📦 增加维度：[bs, slen, heads, 1, dim]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  # 🎈 扩展到重复
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)  # 🔄 合并维度
    )


class Attention(nn.Module):
    """
    👁️ 注意力机制：大脑的"聚光灯系统"
    
    工作原理像学生在图书馆找书：
    1. 生成"问题"(Q)：我要找什么？
    2. 生成"钥匙"(K)：每本书的关键词
    3. 生成"答案"(V)：书的具体内容
    4. 计算匹配度：问题与钥匙的相似度
    5. 提取答案：按匹配度加权提取内容
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 🧮 计算KV头数：如果未指定，默认等于注意力头数
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0  # ⚖️ 必须整除
        
        # 📊 本地计算参数
        self.n_local_heads = args.num_attention_heads  # 本地注意力头数
        self.n_local_kv_heads = self.num_key_value_heads  # 本地KV头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 🔁 每个KV头需要重复的次数
        
        # 📏 每个注意力头的维度：512维/8头 = 64维/头
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # 🎯 线性投影层：像把512维输入翻译成不同语言
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # 💉 Dropout层：像神经元的"随机失忆"防止过拟合
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        # ⚡ 是否使用Flash Attention：像GPU的"涡轮增压"
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        🔄 注意力前向传播：像"聚光灯扫描"过程
        
        参数：
        - x: 输入特征 [batch_size, seq_len, hidden_size]
        - position_embeddings: (cos, sin) 位置编码
        - past_key_value: 缓存的KV，像"记忆"
        - use_cache: 是否使用缓存
        - attention_mask: 注意力掩码，像"眼罩"
        """
        bsz, seq_len, _ = x.shape
        
        # 🎯 步骤1：生成QKV，像把输入翻译成三种语言
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # 🧩 步骤2：重塑形状，像把长纸条切成小方块
        # [batch, seq_len, num_heads * head_dim] → [batch, seq_len, num_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 🌪️ 步骤3：应用旋转位置编码，像给每个位置加上"指纹"
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 🧠 步骤4：KV缓存实现，像"记忆系统"
        if past_key_value is not None:
            # 📚 把新记忆追加到旧记忆：像把新照片加到相册
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None  # 🎯 返回缓存供下次使用

        # 🔄 步骤5：转置维度，像把行变成列
        xq, xk, xv = (
            xq.transpose(1, 2),  # [batch, heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # 🔁 重复KV匹配Q的头数
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # ⚡ Flash Attention路径：像GPU的"快速通道"
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                # 🎭 扩展注意力掩码：像给每个头复制眼罩
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            # 🚀 Flash Attention：GPU优化的矩阵乘法
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            # 🐌 标准注意力：像手工计算匹配度
            # 计算注意力分数：Q·K^T / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 🎭 应用因果掩码：像"只能看前面，不能看后面"
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask
            
            # 🎭 应用注意力掩码：像"选择性眼罩"
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9  # 🎯 把0变成-∞，1变成0
                scores = scores + extended_attention_mask
            
            # 📊 Softmax归一化：像把匹配度转换成概率
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)  # 💉 随机失忆防止过拟合
            output = scores @ xv  # 🎯 提取答案：按概率加权求和

        # 🔄 步骤6：重塑输出，像把小方块拼回长纸条
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))  # 💉 输出投影+dropout
        return output, past_kv


class FeedForward(nn.Module):
    """
    🏭 前馈网络：神经元的"消化工厂"
    
    工作流程像消化食物：
    1. 扩张：512维→1365维（像食物分解成营养分子）
    2. 激活：SiLU选择吸收（像酶决定吸收什么）
    3. 压缩：1365维→512维（像营养重组成身体需要）
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 🧮 计算中间维度：像计算"胃容量"
        if config.intermediate_size is None:
            # 计算：512 * 8/3 ≈ 1365.33，取64的倍数
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # 🎯 三个线性层：像消化系统的三个器官
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 🚪 门控投影
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # ⬇️ 降维投影
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # ⬆️ 升维投影
        
        self.dropout = nn.Dropout(config.dropout)  # 💉 随机失忆
        self.act_fn = ACT2FN[config.hidden_act]  # 🧪 激活函数：SiLU像温和消化酶

    def forward(self, x):
        """
        🔄 前向传播：像"食物消化"过程
        
        计算公式：down_proj(act_fn(gate_proj(x)) * up_proj(x))
        解释：
        1. gate_proj(x) → 门控信号（决定吸收什么）
        2. up_proj(x) → 升维信号（增加表达能力）
        3. act_fn(... * ...) → 激活选择（像酶选择营养）
        4. down_proj → 降维输出（压缩回原始维度）
        """
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    🏥 MoE门控系统：像"医院分诊台"
    
    工作流程：
    1. 症状评估：给每个专家打分
    2. 专家选择：选top-k个专家
    3. 负载均衡：防止某个专家太忙
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 🏆 选择前k个专家
        self.n_routed_experts = config.n_routed_experts  # 👨‍⚕️ 总专家数

        self.scoring_func = config.scoring_func  # 📊 评分函数
        self.alpha = config.aux_loss_alpha  # ⚖️ 负载均衡强度
        self.seq_aux = config.seq_aux  # 📏 序列级辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 📊 是否标准化权重
        self.gating_dim = config.hidden_size  # 🚪 门控维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """🎯 参数初始化：像给专家分配初始能力值"""
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        🔄 门控前向传播：像"分诊"过程
        
        返回：
        - topk_idx: 选择的专家索引
        - topk_weight: 专家权重
        - aux_loss: 负载均衡损失
        """
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # 📏 展平处理
        
        # 📊 计算专家分数：像给每个专家打分
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 🎯 Softmax像投票系统
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 🏆 选择top-k专家：像选最好的k个医生
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 📊 标准化权重：确保总和为1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  # 🛡️ 防除零
            topk_weight = topk_weight / denominator

        # ⚖️ 计算辅助损失：像"工作量平衡"检查
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # 📏 序列级平衡：像按句子统计工作量
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 🎯 token级平衡：像按词统计工作量
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)  # 每个专家的工作量
                Pi = scores_for_aux.mean(0)  # 每个专家的平均分数
                fi = ce * self.n_routed_experts  # 负载因子
                aux_loss = (Pi * fi).sum() * self.alpha  # 负载均衡损失
        else:
            aux_loss = 0  # 🎯 非训练模式无辅助损失
            
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    🏥 专家混合前馈：像"医院会诊系统"
    
    工作流程：
    1. 分诊：门控系统分配专家
    2. 会诊：每个专家独立处理
    3. 汇总：按权重合并专家意见
    4. 补充：全科医生提供通用建议
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        
        # 👨‍⚕️ 创建专家列表：像组建医疗团队
        self.experts = nn.ModuleList([
            FeedForward(config)  # 🏭 每个专家是一个前馈网络
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)  # 🚪 门控系统
        
        # 👨‍⚕️ 共享专家：像全科医生处理基础问题
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        🔄 MoE前向传播：像"专家会诊"过程
        
        1. 分诊：gate(x)决定用哪些专家
        2. 处理：每个专家独立处理分配到的token
        3. 汇总：按权重合并专家输出
        4. 补充：共享专家提供通用处理
        """
        identity = x  # 🎯 保存原始输入（残差连接用）
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 🚪 步骤1：分诊系统选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 📏 步骤2：重塑形状便于处理
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 🎯 训练模式：并行处理所有专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)  # 🔁 复制token
            y = torch.empty_like(x, dtype=torch.float16)  # 📦 预分配输出
            
            # 👨‍⚕️ 每个专家处理分配到的token
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            
            # 📊 加权汇总：像按专家权重合并意见
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 🎯 推理模式：高效处理，只计算用到的专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 👨‍⚕️ 步骤3：共享专家补充处理
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)  # 🎯 残差连接
        
        self.aux_loss = aux_loss  # 📊 保存辅助损失用于训练
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        🚀 高效推理：像"专家门诊"批量处理
        
        优化策略：
        1. 按专家分组token
        2. 批量处理每个专家的所有token
        3. 减少内存碎片和计算冗余
        """
        expert_cache = torch.zeros_like(x)  # 📦 预缓存输出
        idxs = flat_expert_indices.argsort()  # 📊 排序便于分组
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 🎯 批量处理每个专家的所有token
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # 🎯 跳过无token的专家
            
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]  # 📍 该专家处理的token索引
            expert_tokens = x[exp_token_idx]  # 📦 提取token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)  # 🏭 专家处理
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # ⚖️ 应用权重
            
            # 🎯 分散写回：像把处理结果放回正确位置
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    🧱 MiniMind基础块：像"大脑皮层的一个功能区域"
    
    每个块包含：
    1. 注意力系统：像聚光灯决定关注什么
    2. 前馈系统：像消化工厂处理信息
    3. 归一化层：像信号调节器
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # 🎯 核心组件
        self.self_attn = Attention(config)  # 👁️ 注意力系统
        self.layer_id = layer_id
        
        # ⚖️ 归一化层：像信号标准化器
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 🏭 前馈网络：根据配置选择标准或专家混合
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        🔄 块的前向传播：像"大脑皮层的信息处理流水线"
        
        1. 归一化输入：像标准化感官信号
        2. 注意力处理：像聚光灯选择关注信息
        3. 残差连接：像高速公路让信息快速通过
        4. 前馈处理：像深度加工信息
        5. 再次残差连接
        """
        # 🎯 步骤1：注意力子层（带残差连接）
        residual = hidden_states  # 🛣️ 保存原始输入（残差连接）
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # ⚖️ 先归一化
            position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual  # 🛣️ 残差连接：像信息高速公路
        
        # 🎯 步骤2：前馈子层（带残差连接）
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    🧠 MiniMind完整模型：像"8层大脑皮层"
    
    架构：
    1. 嵌入层：像感官接收器
    2. 8层处理：每层像不同深度的脑区
    3. 最终归一化：像意识整合
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        
        # 📊 基本参数
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        
        # 🎯 核心组件
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 📖 词嵌入
        self.dropout = nn.Dropout(config.dropout)  # 💉 输入dropout
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])  # 🧱 8层大脑
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # ⚖️ 最终归一化
        
        # 🌪️ 预计算位置编码：像32768个预设的"位置指纹"
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        🔄 完整前向传播：像"大脑处理语言的完整流程"
        
        1. 嵌入：文字→向量（像感官转换）
        2. 逐层处理：8层大脑皮层加工
        3. 归一化：最终整合输出
        4. 收集辅助损失：用于训练MOE
        """
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 🎯 计算起始位置（用于KV缓存）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        # 📖 步骤1：词嵌入 + Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        # 🌪️ 步骤2：准备位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        
        # 🧱 步骤3：逐层处理（像信息通过8层大脑皮层）
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        
        # ⚖️ 步骤4：最终归一化（像意识整合）
        hidden_states = self.norm(hidden_states)
        
        # 📊 步骤5：收集MOE辅助损失
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    🎯 因果语言模型：像"文字接龙大师"
    
    功能：
    1. 接收前文，预测下一个词
    2. 支持KV缓存加速推理
    3. 支持MOE专家系统
    4. 权重共享减少参数量
    
    权重共享技巧：
    - embed_tokens.weight = lm_head.weight
    - 减少50%参数量，像输入输出共用同一本词典
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # 🧠 核心组件
        self.model = MiniMindModel(self.config)  # 🎯 基础模型
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 🎯 语言模型头
        
        # 🔗 权重共享：像输入输出共用词典
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # 📦 输出包装：像标准格式的响应
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        🔄 语言模型前向传播：像"文字接龙"游戏
        
        1. 编码：文字→向量
        2. 处理：通过8层大脑
        3. 解码：向量→词概率
        4. 输出：标准格式结果
        """
        # 🎯 步骤1：通过基础模型处理
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # 🎯 步骤2：语言模型头解码
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])  # 🎯 预测下一个词概率
        
        # 📦 步骤3：包装输出
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
