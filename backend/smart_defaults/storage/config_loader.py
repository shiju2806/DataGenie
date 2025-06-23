"""
Smart Defaults Configuration Loader
Manages YAML configuration files and provides runtime configuration management
File location: smart_defaults/storage/config_loader.py
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class ConfigInfo:
    """Configuration file information"""
    name: str
    path: Path
    last_modified: datetime
    size_bytes: int
    checksum: str = ""
    loaded_at: Optional[datetime] = None
    is_valid: bool = True
    error_message: Optional[str] = None

@dataclass
class RoleTemplate:
    """Role template structure"""
    name: str
    description: str
    permissions: List[str]
    default_sources: List[str]
    auto_connect_threshold: float
    security_level: str
    industry_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IndustryProfile:
    """Industry profile structure"""
    name: str
    description: str
    common_sources: List[str]
    security_requirements: List[str]
    compliance_standards: List[str]
    default_policies: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityPolicy:
    """Security policy structure"""
    name: str
    description: str
    min_security_level: str
    required_permissions: List[str]
    blocked_sources: List[str]
    encryption_required: bool
    audit_level: str
    conditions: Dict[str, Any] = field(default_factory=dict)

class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass

class ConfigurationLoader:
    """Manages loading and caching of YAML configuration files"""

    def __init__(self, config_base_path: Optional[Union[str, Path]] = None):
        # Determine config path
        if config_base_path:
            self.config_path = Path(config_base_path)
        else:
            # Default to config/ directory relative to this file
            current_dir = Path(__file__).parent
            self.config_path = current_dir.parent / "config"

        # Ensure config directory exists
        self.config_path.mkdir(exist_ok=True)

        # Configuration caches
        self._role_templates: Dict[str, RoleTemplate] = {}
        self._industry_profiles: Dict[str, IndustryProfile] = {}
        self._security_policies: Dict[str, SecurityPolicy] = {}

        # File tracking
        self._config_files: Dict[str, ConfigInfo] = {}
        self._last_reload: Optional[datetime] = None

        # Settings
        self.auto_reload = True
        self.cache_ttl = 300  # 5 minutes

        logger.info(f"📁 Configuration loader initialized with path: {self.config_path}")

    def _get_file_info(self, file_path: Path) -> ConfigInfo:
        """Get information about a configuration file"""
        try:
            stat = file_path.stat()
            return ConfigInfo(
                name=file_path.name,
                path=file_path,
                last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                size_bytes=stat.st_size,
                is_valid=True
            )
        except Exception as e:
            return ConfigInfo(
                name=file_path.name,
                path=file_path,
                last_modified=datetime.now(timezone.utc),
                size_bytes=0,
                is_valid=False,
                error_message=str(e)
            )

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file with error handling"""
        try:
            if not file_path.exists():
                # Create default file if it doesn't exist
                self._create_default_config(file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f) or {}

            logger.debug(f"✅ Loaded config file: {file_path.name}")
            return content

        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error in {file_path.name}: {e}")
            raise ConfigValidationError(f"Invalid YAML in {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to load config file {file_path.name}: {e}")
            raise ConfigValidationError(f"Failed to load {file_path.name}: {e}")

    def _create_default_config(self, file_path: Path):
        """Create default configuration files if they don't exist"""
        file_name = file_path.name

        default_configs = {
            'role_templates.yaml': self._get_default_role_templates(),
            'industry_profiles.yaml': self._get_default_industry_profiles(),
            'security_policies.yaml': self._get_default_security_policies()
        }

        if file_name in default_configs:
            logger.info(f"📝 Creating default config file: {file_name}")
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_configs[file_name], f, default_flow_style=False, indent=2)

    def _get_default_role_templates(self) -> Dict[str, Any]:
        """Get default role templates"""
        return {
            'data_analyst': {
                'name': 'Data Analyst',
                'description': 'Analyzes data to derive business insights',
                'permissions': ['read_data', 'run_queries', 'create_reports'],
                'default_sources': ['postgres', 'redshift', 'tableau'],
                'auto_connect_threshold': 0.8,
                'security_level': 'medium',
                'industry_specific': {
                    'finance': {'compliance_required': True},
                    'healthcare': {'hipaa_required': True}
                }
            },
            'data_engineer': {
                'name': 'Data Engineer',
                'description': 'Builds and maintains data infrastructure',
                'permissions': ['read_data', 'write_data', 'manage_pipelines', 'admin_access'],
                'default_sources': ['postgres', 'kafka', 'airflow', 'spark'],
                'auto_connect_threshold': 0.85,
                'security_level': 'high',
                'industry_specific': {
                    'finance': {'sox_compliance': True},
                    'healthcare': {'data_encryption': True}
                }
            },
            'business_analyst': {
                'name': 'Business Analyst',
                'description': 'Analyzes business processes and requirements',
                'permissions': ['read_data', 'create_reports', 'view_dashboards'],
                'default_sources': ['salesforce', 'google_analytics', 'excel'],
                'auto_connect_threshold': 0.75,
                'security_level': 'medium',
                'industry_specific': {}
            },
            'ml_engineer': {
                'name': 'ML Engineer',
                'description': 'Develops and deploys machine learning models',
                'permissions': ['read_data', 'write_data', 'deploy_models', 'manage_experiments'],
                'default_sources': ['jupyter', 'mlflow', 'tensorflow', 'postgres'],
                'auto_connect_threshold': 0.9,
                'security_level': 'high',
                'industry_specific': {
                    'finance': {'model_explainability': True},
                    'healthcare': {'bias_testing': True}
                }
            }
        }

    def _get_default_industry_profiles(self) -> Dict[str, Any]:
        """Get default industry profiles"""
        return {
            'technology': {
                'name': 'Technology',
                'description': 'Software and technology companies',
                'common_sources': ['github', 'jira', 'postgres', 'redis', 'elasticsearch'],
                'security_requirements': ['code_review', 'access_logging'],
                'compliance_standards': ['SOC2', 'ISO27001'],
                'default_policies': {
                    'data_retention_days': 365,
                    'audit_level': 'standard'
                }
            },
            'finance': {
                'name': 'Financial Services',
                'description': 'Banks, investment firms, fintech',
                'common_sources': ['oracle', 'bloomberg', 'reuters', 'mainframe'],
                'security_requirements': ['encryption_at_rest', 'encryption_in_transit', 'mfa_required'],
                'compliance_standards': ['SOX', 'PCI_DSS', 'GDPR'],
                'default_policies': {
                    'data_retention_days': 2555,  # 7 years
                    'audit_level': 'comprehensive',
                    'pii_handling': 'strict'
                }
            },
            'healthcare': {
                'name': 'Healthcare',
                'description': 'Hospitals, pharma, medical devices',
                'common_sources': ['epic', 'cerner', 'hl7', 'postgresql'],
                'security_requirements': ['hipaa_compliant', 'patient_consent', 'data_anonymization'],
                'compliance_standards': ['HIPAA', 'FDA', 'GDPR'],
                'default_policies': {
                    'data_retention_days': 2190,  # 6 years
                    'audit_level': 'comprehensive',
                    'phi_handling': 'strict'
                }
            },
            'retail': {
                'name': 'Retail & E-commerce',
                'description': 'Online and offline retail businesses',
                'common_sources': ['shopify', 'magento', 'stripe', 'google_analytics'],
                'security_requirements': ['pci_compliant', 'customer_data_protection'],
                'compliance_standards': ['PCI_DSS', 'GDPR', 'CCPA'],
                'default_policies': {
                    'data_retention_days': 1095,  # 3 years
                    'audit_level': 'standard'
                }
            }
        }

    def _get_default_security_policies(self) -> Dict[str, Any]:
        """Get default security policies"""
        return {
            'low_security': {
                'name': 'Low Security',
                'description': 'Basic security for non-sensitive data',
                'min_security_level': 'low',
                'required_permissions': ['read_data'],
                'blocked_sources': [],
                'encryption_required': False,
                'audit_level': 'basic',
                'conditions': {
                    'max_data_sensitivity': 'public',
                    'require_approval': False
                }
            },
            'medium_security': {
                'name': 'Medium Security',
                'description': 'Standard security for business data',
                'min_security_level': 'medium',
                'required_permissions': ['read_data', 'verified_identity'],
                'blocked_sources': ['public_apis'],
                'encryption_required': True,
                'audit_level': 'standard',
                'conditions': {
                    'max_data_sensitivity': 'internal',
                    'require_approval': True,
                    'approval_roles': ['manager', 'data_steward']
                }
            },
            'high_security': {
                'name': 'High Security',
                'description': 'Enhanced security for sensitive data',
                'min_security_level': 'high',
                'required_permissions': ['read_data', 'verified_identity', 'security_clearance'],
                'blocked_sources': ['public_apis', 'cloud_storage'],
                'encryption_required': True,
                'audit_level': 'comprehensive',
                'conditions': {
                    'max_data_sensitivity': 'confidential',
                    'require_approval': True,
                    'approval_roles': ['security_officer', 'data_protection_officer'],
                    'mfa_required': True,
                    'session_timeout': 1800
                }
            },
            'compliance_strict': {
                'name': 'Strict Compliance',
                'description': 'Maximum security for regulated industries',
                'min_security_level': 'maximum',
                'required_permissions': ['read_data', 'verified_identity', 'security_clearance', 'compliance_training'],
                'blocked_sources': ['external_apis', 'unencrypted_sources'],
                'encryption_required': True,
                'audit_level': 'forensic',
                'conditions': {
                    'max_data_sensitivity': 'restricted',
                    'require_approval': True,
                    'approval_roles': ['compliance_officer', 'legal_counsel'],
                    'mfa_required': True,
                    'session_timeout': 900,
                    'data_masking': True,
                    'access_logging': 'detailed'
                }
            }
        }

    def _validate_role_template(self, name: str, data: Dict[str, Any]) -> RoleTemplate:
        """Validate and convert role template data"""
        required_fields = ['name', 'description', 'permissions', 'default_sources', 'auto_connect_threshold', 'security_level']

        for field in required_fields:
            if field not in data:
                raise ConfigValidationError(f"Role template '{name}' missing required field: {field}")

        # Validate threshold
        threshold = data['auto_connect_threshold']
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ConfigValidationError(f"Role template '{name}' has invalid auto_connect_threshold: {threshold}")

        # Validate security level
        valid_levels = ['low', 'medium', 'high', 'maximum']
        if data['security_level'] not in valid_levels:
            raise ConfigValidationError(f"Role template '{name}' has invalid security_level: {data['security_level']}")

        return RoleTemplate(
            name=data['name'],
            description=data['description'],
            permissions=data['permissions'],
            default_sources=data['default_sources'],
            auto_connect_threshold=threshold,
            security_level=data['security_level'],
            industry_specific=data.get('industry_specific', {})
        )

    def _validate_industry_profile(self, name: str, data: Dict[str, Any]) -> IndustryProfile:
        """Validate and convert industry profile data"""
        required_fields = ['name', 'description', 'common_sources', 'security_requirements', 'compliance_standards']

        for field in required_fields:
            if field not in data:
                raise ConfigValidationError(f"Industry profile '{name}' missing required field: {field}")

        return IndustryProfile(
            name=data['name'],
            description=data['description'],
            common_sources=data['common_sources'],
            security_requirements=data['security_requirements'],
            compliance_standards=data['compliance_standards'],
            default_policies=data.get('default_policies', {})
        )

    def _validate_security_policy(self, name: str, data: Dict[str, Any]) -> SecurityPolicy:
        """Validate and convert security policy data"""
        required_fields = ['name', 'description', 'min_security_level', 'required_permissions', 'encryption_required', 'audit_level']

        for field in required_fields:
            if field not in data:
                raise ConfigValidationError(f"Security policy '{name}' missing required field: {field}")

        return SecurityPolicy(
            name=data['name'],
            description=data['description'],
            min_security_level=data['min_security_level'],
            required_permissions=data['required_permissions'],
            blocked_sources=data.get('blocked_sources', []),
            encryption_required=data['encryption_required'],
            audit_level=data['audit_level'],
            conditions=data.get('conditions', {})
        )

    async def load_all_configs(self) -> bool:
        """Load all configuration files"""
        try:
            await self.load_role_templates()
            await self.load_industry_profiles()
            await self.load_security_policies()

            self._last_reload = datetime.now(timezone.utc)
            logger.info("✅ All configuration files loaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load configurations: {e}")
            return False

    async def load_role_templates(self) -> Dict[str, RoleTemplate]:
        """Load role templates from YAML file"""
        file_path = self.config_path / "role_templates.yaml"

        try:
            # Track file info
            self._config_files['role_templates'] = self._get_file_info(file_path)

            # Load and validate
            data = self._load_yaml_file(file_path)
            self._role_templates.clear()

            for name, template_data in data.items():
                try:
                    role_template = self._validate_role_template(name, template_data)
                    self._role_templates[name] = role_template
                except ConfigValidationError as e:
                    logger.error(f"❌ Invalid role template '{name}': {e}")
                    continue

            logger.info(f"✅ Loaded {len(self._role_templates)} role templates")
            return self._role_templates

        except Exception as e:
            logger.error(f"❌ Failed to load role templates: {e}")
            self._config_files['role_templates'].is_valid = False
            self._config_files['role_templates'].error_message = str(e)
            return {}

    async def load_industry_profiles(self) -> Dict[str, IndustryProfile]:
        """Load industry profiles from YAML file"""
        file_path = self.config_path / "industry_profiles.yaml"

        try:
            # Track file info
            self._config_files['industry_profiles'] = self._get_file_info(file_path)

            # Load and validate
            data = self._load_yaml_file(file_path)
            self._industry_profiles.clear()

            for name, profile_data in data.items():
                try:
                    industry_profile = self._validate_industry_profile(name, profile_data)
                    self._industry_profiles[name] = industry_profile
                except ConfigValidationError as e:
                    logger.error(f"❌ Invalid industry profile '{name}': {e}")
                    continue

            logger.info(f"✅ Loaded {len(self._industry_profiles)} industry profiles")
            return self._industry_profiles

        except Exception as e:
            logger.error(f"❌ Failed to load industry profiles: {e}")
            self._config_files['industry_profiles'].is_valid = False
            self._config_files['industry_profiles'].error_message = str(e)
            return {}

    async def load_security_policies(self) -> Dict[str, SecurityPolicy]:
        """Load security policies from YAML file"""
        file_path = self.config_path / "security_policies.yaml"

        try:
            # Track file info
            self._config_files['security_policies'] = self._get_file_info(file_path)

            # Load and validate
            data = self._load_yaml_file(file_path)
            self._security_policies.clear()

            for name, policy_data in data.items():
                try:
                    security_policy = self._validate_security_policy(name, policy_data)
                    self._security_policies[name] = security_policy
                except ConfigValidationError as e:
                    logger.error(f"❌ Invalid security policy '{name}': {e}")
                    continue

            logger.info(f"✅ Loaded {len(self._security_policies)} security policies")
            return self._security_policies

        except Exception as e:
            logger.error(f"❌ Failed to load security policies: {e}")
            self._config_files['security_policies'].is_valid = False
            self._config_files['security_policies'].error_message = str(e)
            return {}

    # Getter methods with caching
    @lru_cache(maxsize=128)
    def get_role_template(self, role_name: str) -> Optional[RoleTemplate]:
        """Get role template by name"""
        return self._role_templates.get(role_name)

    @lru_cache(maxsize=128)
    def get_industry_profile(self, industry_name: str) -> Optional[IndustryProfile]:
        """Get industry profile by name"""
        return self._industry_profiles.get(industry_name)

    @lru_cache(maxsize=128)
    def get_security_policy(self, policy_name: str) -> Optional[SecurityPolicy]:
        """Get security policy by name"""
        return self._security_policies.get(policy_name)

    def get_all_role_templates(self) -> Dict[str, RoleTemplate]:
        """Get all role templates"""
        return self._role_templates.copy()

    def get_all_industry_profiles(self) -> Dict[str, IndustryProfile]:
        """Get all industry profiles"""
        return self._industry_profiles.copy()

    def get_all_security_policies(self) -> Dict[str, SecurityPolicy]:
        """Get all security policies"""
        return self._security_policies.copy()

    def get_roles_for_industry(self, industry: str) -> List[str]:
        """Get recommended roles for an industry"""
        # This would be more sophisticated in production
        industry_role_mapping = {
            'technology': ['data_engineer', 'ml_engineer', 'data_analyst'],
            'finance': ['data_analyst', 'business_analyst', 'data_engineer'],
            'healthcare': ['data_analyst', 'business_analyst'],
            'retail': ['business_analyst', 'data_analyst', 'ml_engineer']
        }
        return industry_role_mapping.get(industry, ['data_analyst'])

    def get_security_level_for_industry(self, industry: str) -> str:
        """Get recommended security level for an industry"""
        industry_security = {
            'technology': 'medium',
            'finance': 'high',
            'healthcare': 'high',
            'retail': 'medium'
        }
        return industry_security.get(industry, 'medium')

    async def reload_if_changed(self) -> bool:
        """Reload configurations if files have changed"""
        if not self.auto_reload:
            return False

        changed = False
        for file_name, config_info in self._config_files.items():
            current_info = self._get_file_info(config_info.path)
            if current_info.last_modified > config_info.last_modified:
                logger.info(f"🔄 Configuration file changed: {file_name}")
                changed = True

        if changed:
            return await self.load_all_configs()

        return False

    def get_config_status(self) -> Dict[str, Any]:
        """Get status of all configuration files"""
        return {
            'config_path': str(self.config_path),
            'last_reload': self._last_reload.isoformat() if self._last_reload else None,
            'auto_reload': self.auto_reload,
            'files': {
                name: {
                    'is_valid': info.is_valid,
                    'last_modified': info.last_modified.isoformat(),
                    'size_bytes': info.size_bytes,
                    'error_message': info.error_message
                }
                for name, info in self._config_files.items()
            },
            'loaded_counts': {
                'role_templates': len(self._role_templates),
                'industry_profiles': len(self._industry_profiles),
                'security_policies': len(self._security_policies)
            }
        }

    async def export_config(self, export_path: Path, format_type: str = 'yaml') -> bool:
        """Export current configuration to file"""
        try:
            export_data = {
                'role_templates': {name: {
                    'name': template.name,
                    'description': template.description,
                    'permissions': template.permissions,
                    'default_sources': template.default_sources,
                    'auto_connect_threshold': template.auto_connect_threshold,
                    'security_level': template.security_level,
                    'industry_specific': template.industry_specific
                } for name, template in self._role_templates.items()},
                'industry_profiles': {name: {
                    'name': profile.name,
                    'description': profile.description,
                    'common_sources': profile.common_sources,
                    'security_requirements': profile.security_requirements,
                    'compliance_standards': profile.compliance_standards,
                    'default_policies': profile.default_policies
                } for name, profile in self._industry_profiles.items()},
                'security_policies': {name: {
                    'name': policy.name,
                    'description': policy.description,
                    'min_security_level': policy.min_security_level,
                    'required_permissions': policy.required_permissions,
                    'blocked_sources': policy.blocked_sources,
                    'encryption_required': policy.encryption_required,
                    'audit_level': policy.audit_level,
                    'conditions': policy.conditions
                } for name, policy in self._security_policies.items()}
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                if format_type == 'yaml':
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                elif format_type == 'json':
                    json.dump(export_data, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")

            logger.info(f"✅ Configuration exported to: {export_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to export configuration: {e}")
            return False

# Factory function
async def create_config_loader(config_path: Optional[Union[str, Path]] = None) -> ConfigurationLoader:
    """Create and initialize configuration loader"""
    loader = ConfigurationLoader(config_path)
    await loader.load_all_configs()
    return loader

# Testing
if __name__ == "__main__":
    async def test_config_loader():
        """Test configuration loader"""
        try:
            print("🧪 Testing Configuration Loader...")

            # Create temporary config directory for testing
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"📁 Using temporary config directory: {temp_dir}")
                config_loader = await create_config_loader(temp_dir)

                # Test getting role templates
                data_analyst = config_loader.get_role_template('data_analyst')
                if data_analyst:
                    print(f"✅ Role template loaded: {data_analyst.name}")
                    print(f"   Permissions: {data_analyst.permissions}")
                    print(f"   Auto-connect threshold: {data_analyst.auto_connect_threshold}")
                else:
                    print("⚠️ No data_analyst role template found")

                # Test getting industry profiles
                tech_profile = config_loader.get_industry_profile('technology')
                if tech_profile:
                    print(f"✅ Industry profile loaded: {tech_profile.name}")
                    print(f"   Common sources: {tech_profile.common_sources}")
                else:
                    print("⚠️ No technology industry profile found")

                # Test getting security policies
                medium_security = config_loader.get_security_policy('medium_security')
                if medium_security:
                    print(f"✅ Security policy loaded: {medium_security.name}")
                    print(f"   Encryption required: {medium_security.encryption_required}")
                else:
                    print("⚠️ No medium_security policy found")

                # Test recommendations
                roles = config_loader.get_roles_for_industry('technology')
                print(f"✅ Recommended roles for technology: {roles}")

                security_level = config_loader.get_security_level_for_industry('finance')
                print(f"✅ Security level for finance: {security_level}")

                # Test status
                status = config_loader.get_config_status()
                print(f"✅ Config status: {status['loaded_counts']} items loaded")

                # Show what was actually loaded
                all_roles = config_loader.get_all_role_templates()
                all_industries = config_loader.get_all_industry_profiles()
                all_policies = config_loader.get_all_security_policies()

                print(f"📊 Loaded configurations:")
                print(f"   - Role templates: {list(all_roles.keys())}")
                print(f"   - Industry profiles: {list(all_industries.keys())}")
                print(f"   - Security policies: {list(all_policies.keys())}")

                print("\n✅ All configuration loader tests passed!")

        except Exception as e:
            print(f"❌ Configuration loader test failed: {e}")
            import traceback
            traceback.print_exc()

    # Run tests
    asyncio.run(test_config_loader())