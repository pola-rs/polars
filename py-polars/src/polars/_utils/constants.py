from datetime import date, datetime, timezone
from typing import Final

# Integer ranges
I8_MIN: Final = -(2**7)
I16_MIN: Final = -(2**15)
I32_MIN: Final = -(2**31)
I64_MIN: Final = -(2**63)
I128_MIN: Final = -(2**127)
I8_MAX: Final = 2**7 - 1
I16_MAX: Final = 2**15 - 1
I32_MAX: Final = 2**31 - 1
I64_MAX: Final = 2**63 - 1
I128_MAX: Final = 2**127 - 1
U8_MAX: Final = 2**8 - 1
U16_MAX: Final = 2**16 - 1
U32_MAX: Final = 2**32 - 1
U64_MAX: Final = 2**64 - 1
U128_MAX: Final = 2**128 - 1

# Temporal
SECONDS_PER_DAY: Final = 86_400
SECONDS_PER_HOUR: Final = 3_600
NS_PER_SECOND: Final = 1_000_000_000
US_PER_SECOND: Final = 1_000_000
MS_PER_SECOND: Final = 1_000

EPOCH_DATE: Final = date(1970, 1, 1)
EPOCH: Final = datetime(1970, 1, 1).replace(tzinfo=None)
EPOCH_UTC: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)
