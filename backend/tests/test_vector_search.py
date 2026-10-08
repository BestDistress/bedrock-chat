import unittest
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

from app.repositories.models.custom_bot_kb import (
    BedrockAgentGetKnowledgeBaseResponse,
    KnowledgeBase,
    KnowledgeBaseConfiguration,
)
from app.repositories.knowledge_base import get_knowledge_base_info
from app.vector_search import _bedrock_knowledge_base_search, agent_client
from botocore.validate import ParamValidator


def create_bot(knowledge_base_type: Literal["dedicated", "shared"]):
    return SimpleNamespace(
        id="bot-1",
        bedrock_knowledge_base=SimpleNamespace(
            type=knowledge_base_type,
            knowledge_base_id="kb12345678",
            exist_knowledge_base_id=None,
            search_params=SimpleNamespace(max_results=7, search_type="hybrid"),
        ),
    )


def create_knowledge_base_info(
    knowledge_base_type: Literal["VECTOR", "KENDRA", "SQL", "MANAGED"],
) -> BedrockAgentGetKnowledgeBaseResponse:
    return BedrockAgentGetKnowledgeBaseResponse(
        knowledge_base=KnowledgeBase(
            knowledge_base_configuration=KnowledgeBaseConfiguration(
                type=knowledge_base_type,
            ),
        ),
    )


class TestBedrockKnowledgeBaseSearch(unittest.TestCase):
    def test_runtime_sdk_accepts_managed_search_configuration(self):
        operation_model = agent_client.meta.service_model.operation_model("Retrieve")
        parameters = {
            "knowledgeBaseId": "kb12345678",
            "retrievalQuery": {"text": "test query"},
            "retrievalConfiguration": {
                "managedSearchConfiguration": {"numberOfResults": 7},
            },
        }

        validation_report = ParamValidator().validate(
            parameters,
            operation_model.input_shape,
        )

        self.assertFalse(
            validation_report.has_errors(),
            validation_report.generate_report(),
        )

    @patch("app.repositories.knowledge_base.get_bedrock_agent_client")
    def test_get_knowledge_base_info_accepts_managed_type(self, get_client):
        client = get_client.return_value
        client.get_knowledge_base.return_value = {
            "knowledgeBase": {
                "knowledgeBaseConfiguration": {"type": "MANAGED"},
            },
        }

        result = get_knowledge_base_info("kb12345678")

        self.assertEqual(
            result.knowledge_base.knowledge_base_configuration.type,
            "MANAGED",
        )

    @patch("app.vector_search.get_knowledge_base_info")
    @patch("app.vector_search.agent_client.retrieve", return_value={})
    def test_managed_knowledge_base_uses_managed_search_configuration(
        self, retrieve, get_knowledge_base_info
    ):
        get_knowledge_base_info.return_value = create_knowledge_base_info("MANAGED")

        _bedrock_knowledge_base_search(create_bot("dedicated"), "test query")

        retrieve.assert_called_once_with(
            knowledgeBaseId="kb12345678",
            retrievalQuery={"text": "test query"},
            retrievalConfiguration={
                "managedSearchConfiguration": {"numberOfResults": 7},
            },
        )

    @patch("app.vector_search.get_knowledge_base_info")
    @patch("app.vector_search.agent_client.retrieve", return_value={})
    def test_shared_managed_knowledge_base_keeps_tenant_filter(
        self, retrieve, get_knowledge_base_info
    ):
        get_knowledge_base_info.return_value = create_knowledge_base_info("MANAGED")

        _bedrock_knowledge_base_search(create_bot("shared"), "test query")

        self.assertEqual(
            retrieve.call_args.kwargs["retrievalConfiguration"],
            {
                "managedSearchConfiguration": {
                    "numberOfResults": 7,
                    "filter": {
                        "listContains": {
                            "key": "tenants",
                            "value": "BOT#bot-1",
                        },
                    },
                },
            },
        )

    @patch("app.vector_search.get_knowledge_base_info")
    @patch("app.vector_search.agent_client.retrieve", return_value={})
    def test_vector_knowledge_base_keeps_vector_search_configuration(
        self, retrieve, get_knowledge_base_info
    ):
        get_knowledge_base_info.return_value = create_knowledge_base_info("VECTOR")

        _bedrock_knowledge_base_search(create_bot("dedicated"), "test query")

        retrieve.assert_called_once_with(
            knowledgeBaseId="kb12345678",
            retrievalQuery={"text": "test query"},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 7,
                    "overrideSearchType": "HYBRID",
                },
            },
        )

    @patch("app.vector_search.get_knowledge_base_info")
    @patch("app.vector_search.agent_client.retrieve", return_value={})
    def test_kendra_knowledge_base_omits_override_search_type(
        self, retrieve, get_knowledge_base_info
    ):
        get_knowledge_base_info.return_value = create_knowledge_base_info("KENDRA")

        _bedrock_knowledge_base_search(create_bot("dedicated"), "test query")

        retrieve.assert_called_once_with(
            knowledgeBaseId="kb12345678",
            retrievalQuery={"text": "test query"},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 7},
            },
        )
