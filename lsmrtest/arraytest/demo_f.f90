
subroutine func0()
  write(*,*) "selected_real_kind(15) =>", selected_real_kind(15)
end subroutine func0

subroutine func1(n, CallBack)

  integer :: n
  real(8) :: x(n)

  interface
    subroutine CallBack(n, v)
      integer :: n
      real(8) :: v(n)
    end subroutine CallBack
  end interface

  do i = 1,n
    x(i) = 11 * i
  end do

  call CallBack(n, x)

end subroutine func1

subroutine func2(n, v)

  integer :: n
  real(8) :: v(n)
  real(8) :: w(n)

  do i = 1,n
    w(i) = 101 * i
  end do

  do i = 1,n
    write(*,*) i, w(i), v(i) 
  end do

end subroutine func2

