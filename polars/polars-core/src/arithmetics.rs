pub trait TryAdd<Rhs = Self> {
    type Output;
    type Error;
    fn try_add(self, rhs: Rhs) -> Result<Self::Output, Self::Error>;
}

pub trait TrySub<Rhs = Self> {
    type Output;
    type Error;
    fn try_sub(self, rhs: Rhs) -> Result<Self::Output, Self::Error>;
}

pub trait TryMul<Rhs = Self> {
    type Output;
    type Error;
    fn try_mul(self, rhs: Rhs) -> Result<Self::Output, Self::Error>;
}

pub trait TryDiv<Rhs = Self> {
    type Output;
    type Error;
    fn try_div(self, rhs: Rhs) -> Result<Self::Output, Self::Error>;
}

pub trait TryRem<Rhs = Self> {
    type Output;
    type Error;
    fn try_rem(self, rhs: Rhs) -> Result<Self::Output, Self::Error>;
}
