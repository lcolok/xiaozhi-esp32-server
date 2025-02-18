import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    @staticmethod
    def load_env() -> None:
        """加载环境变量"""
        # 首先尝试加载 .env 文件
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
        
        # 如果存在 data/.env，优先使用它的配置
        data_env_path = Path("data/.env")
        if data_env_path.exists():
            load_dotenv(data_env_path, override=True)

    @staticmethod
    def _replace_env_vars(value: Any) -> Any:
        """递归替换配置中的环境变量"""
        if isinstance(value, str):
            # 匹配 ${VAR} 或 $VAR 格式的环境变量
            pattern = r'\${([^}]+)}|\$([A-Za-z0-9_]+)'
            
            def replace_var(match):
                var_name = match.group(1) or match.group(2)
                return os.environ.get(var_name, f"${{{var_name}}}")
            
            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: ConfigLoader._replace_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [ConfigLoader._replace_env_vars(item) for item in value]
        return value

    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict:
        """加载配置文件"""
        # 首先加载环境变量
        ConfigLoader.load_env()
        
        # 加载主配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查是否存在 data/.config.yaml
        data_config_path = Path("data/.config.yaml")
        if data_config_path.exists():
            with open(data_config_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                # 递归更新配置
                config = ConfigLoader._deep_update(config, data_config)
        
        # 替换环境变量
        config = ConfigLoader._replace_env_vars(config)
        
        return config

    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        """递归更新字典"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                base_dict[key] = ConfigLoader._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
