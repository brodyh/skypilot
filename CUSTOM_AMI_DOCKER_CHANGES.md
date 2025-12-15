# Custom Base AMI with Docker Support

## Summary

Added support for specifying a custom base AMI when using Docker images on AWS. This allows you to use an AMI that already has Docker images pre-loaded, avoiding lengthy docker pulls on every launch.

## Syntax

```yaml
resources:
  cloud: aws
  region: us-west-2
  image_id: ami-0f666d43d12ebc3cc:docker:312471575300.dkr.ecr.us-west-2.amazonaws.com/carla:0.9.16
```

Format: `<base-ami>:docker:<docker-image>`

The delimiter `:docker:` separates the base AMI from the Docker image name.

## Implementation

### Files Modified

1. **sky/resources.py** (1 change)
   - `extract_docker_image()` method - Added support for composite format

2. **sky/clouds/aws.py** (3 changes)
   - `_get_image_id()` - Strips `:docker:...` suffix to extract base AMI
   - `get_image_size()` - Handles composite format before AWS API calls
   - `get_image_root_device_name()` - Handles composite format before AWS API calls

3. **sky/task.py** (1 change)
   - `_with_docker_login_config()` - Preserves composite format when adding Docker credentials

### Code Changes

#### 1. sky/resources.py - `extract_docker_image()` (lines 1260-1276)

```python
def extract_docker_image(self) -> Optional[str]:
    if self.image_id is None:
        return None
    if len(self.image_id) == 1:
        image_key = list(self.image_id.keys())[0]
        if image_key == self.region or image_key is None:
            image_id = self.image_id[image_key]
            # Check for composite format: base-ami:docker:image
            if ':docker:' in image_id:
                # Split on ':docker:' and return everything after
                return image_id.split(':docker:', 1)[1]
            # Legacy format: docker:image
            if image_id.startswith('docker:'):
                return image_id[len('docker:'):]
    return None
```

#### 2. sky/clouds/aws.py - `_get_image_id()` (lines 410-414)

```python
# Strip :docker:... suffix if present (composite format)
# This allows specifying: ami-xxx:docker:image for custom base AMI
if ':docker:' in image_id_str:
    image_id_str = image_id_str.split(':docker:', 1)[0]
    logger.info(f'Using custom base AMI for Docker: {image_id_str}')
```

#### 3. sky/clouds/aws.py - `get_image_size()` (lines 431-436)

```python
# Strip :docker:... suffix if present (composite format ami-xxx:docker:yyy)
if ':docker:' in image_id:
    image_id = image_id.split(':docker:', 1)[0]
# Also handle legacy docker: prefix format
elif image_id.startswith('docker:'):
    image_id = image_id.split('docker:', 1)[1]
```

#### 4. sky/clouds/aws.py - `get_image_root_device_name()` (lines 469-471)

```python
# Strip :docker:... suffix if present (composite format)
if ':docker:' in image_id:
    image_id = image_id.split(':docker:', 1)[0]
```

#### 5. sky/task.py - `_with_docker_login_config()` (lines 201-210)

```python
region = list(resources.image_id.keys())[0]
original_image_id = resources.image_id[region]

# Preserve composite format (ami-xxx:docker:yyy) if present
# Otherwise use legacy format (docker:yyy)
if ':docker:' in original_image_id:
    # Composite format already has base AMI, keep it as-is
    new_image_id = original_image_id
else:
    # Legacy format - reconstruct with docker: prefix
    new_image_id = 'docker:' + docker_image

return resources.copy(image_id={region: new_image_id},
                      _docker_login_config=docker_login_config)
```

## Backward Compatibility

All existing formats continue to work:
- ✅ `image_id: docker:xxx` - Legacy Docker format (uses default SkyPilot AMI)
- ✅ `image_id: ami-xxx` - Regular AMI (no Docker)
- ✅ `image_id: {us-west-2: ami-xxx}` - Region-specific AMI
- ✅ **New**: `image_id: ami-xxx:docker:yyy` - Custom base AMI with Docker

## Testing

Tested with ECR Docker images and custom AWS AMIs. The implementation correctly:
1. Extracts the base AMI from composite format
2. Extracts the Docker image name from composite format
3. Launches instances with the specified custom AMI
4. Starts Docker containers with the specified image
5. Handles Docker authentication credentials
