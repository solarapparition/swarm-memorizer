"""Artifact management."""

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self, Sequence

from swarm_memorizer.schema import NONE, ArtifactType
from swarm_memorizer.toolkit.text import dedent_and_strip


class ArtifactValidationMessage(Enum):
    """Messages for validating an artifact."""

    INVALID_ARTIFACT_TYPE = "Invalid artifact type."
    INLINE_MUST_BE_CREATED = "Inline artifact must be created."
    REMOTE_RESOURCE_MUST_NOT_BE_CREATED = (
        "Remote resource artifact must not be created."
    )
    MISSING_CONTENT = "Content is missing for artifact that must be created."
    MISSING_LOCATION = "Location is missing for artifact that has already been created."


@dataclass
class ArtifactValidationError(Exception):
    """Error when validating an artifact."""

    message: ArtifactValidationMessage
    """Message for the error."""
    artifact: "Artifact"
    """Field values for the artifact that generated the validation error."""


@dataclass
class Artifact:
    """Entry for an artifact."""

    type: ArtifactType
    description: str
    location: str | None
    must_be_created: bool
    content: str | None

    @classmethod
    def from_serialized_data(cls, data: dict[str, Any]) -> Self:
        """Deserialize the artifact from a JSON-compatible dictionary."""
        return cls(
            type=ArtifactType(data["type"]),
            description=data["description"],
            location=data["location"],
            must_be_created=data["must_be_created"],
            content=data["content"],
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the artifact to a JSON-compatible dictionary."""
        serialized_data = asdict(self)
        serialized_data["type"] = self.type.value
        return serialized_data

    def validate(self) -> Literal[True]:
        """Validate the artifact."""
        try:
            assert (
                self.type in ArtifactType
            ), ArtifactValidationMessage.INVALID_ARTIFACT_TYPE
            if self.type == ArtifactType.INLINE:
                assert (
                    self.must_be_created
                ), ArtifactValidationMessage.INLINE_MUST_BE_CREATED
            if self.type == ArtifactType.REMOTE_RESOURCE:
                assert (
                    not self.must_be_created
                ), ArtifactValidationMessage.REMOTE_RESOURCE_MUST_NOT_BE_CREATED
            if self.must_be_created:
                assert self.content, ArtifactValidationMessage.MISSING_CONTENT
            else:
                assert self.location, ArtifactValidationMessage.MISSING_LOCATION
        except AssertionError as error:
            raise ArtifactValidationError(
                message=error.args[0], artifact=self
            ) from error
        return True

    def __str__(self) -> str:
        """String representation of the artifact."""
        # this must be an accurate programmatic representation of the artifact—may be read in to recreate the artifact
        template = """
        - description: {description}
          type: {type}
          location: {location}
          content: {content}
          must_be_created: {must_be_created}
        """
        return dedent_and_strip(template).format(
            description=self.description,
            location=self.location,
            content=self.content,
            type=self.type,
            must_be_created=self.must_be_created,
        )


def artifacts_printout(artifacts: Sequence[Artifact]) -> str:
    """String representation of the artifacts."""
    return "\n".join(str(artifact) for artifact in artifacts) or NONE

def write_file_artifact(artifact: Artifact, output_dir: Path) -> Artifact:
    """Write the content of a file artifact to the output directory. Returns an updated artifact with the location set to the output directory."""
    raise NotImplementedError("TODO")
    # assert artifact.type == ArtifactType.FILE and artifact.content and artifact.must_be_created
    # file_name = f"{convert_to_filename(artifact.description)}.txt"
    # output_path = output_dir / file_name
    # task.output_dir.write_text(artifact.content, encoding="utf-8")
    # return Artifact(
    #     type=artifact.type,
    #     description=artifact.description,
    #     location=str(task.output_dir),
    #     must_be_created=artifact.must_be_created,
    #     content=None,
    # )

