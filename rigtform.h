#ifndef RIGTFORM_H
#define RIGTFORM_H

#include <iostream>
#include <cassert>

#include "matrix4.h"
#include "quat.h"

class RigTForm {
  Cvec3 t_; // translation component
  Quat r_;  // rotation component represented as a quaternion

public:
  RigTForm() : t_(0) {
    assert(norm2(Quat(1,0,0,0) - r_) < CS175_EPS2);
  }

  RigTForm(const Cvec3& t, const Quat& r) {
      t_ = t;
      r_ = r;
  }

  explicit RigTForm(const Cvec3& t) {
      t_ = t;
      r_ = Quat();
  }

  explicit RigTForm(const Quat& r) {
      t_ = Cvec3();
      r_ = r;
      
  }

  Cvec3 getTranslation() const {
    return t_;
  }

  Quat getRotation() const {
    return r_;
  }

  RigTForm& setTranslation(const Cvec3& t) {
    t_ = t;
    return *this;
  }

  RigTForm& setRotation(const Quat& r) {
    r_ = r;
    return *this;
  }

  Cvec4 operator * (const Cvec4& a) const {
    Cvec3 result = r_ * Cvec3(a);
    if (a(3) == 1) {
    	result = result + t_;
    }
    return Cvec4(result, a(3));
  }

  RigTForm operator * (const RigTForm& a) const {
    RigTForm result = RigTForm();
    result.setTranslation(t_ + (r_ * a.t_));
    result.setRotation(r_ * a.r_);
    return result;
  }
};

inline RigTForm inv(const RigTForm& tform) {
    RigTForm result = RigTForm();
    result.setTranslation(-(inv(tform.getRotation()) * tform.getTranslation()));
    result.setRotation(inv(tform.getRotation()));
    return result;
}

inline RigTForm transFact(const RigTForm& tform) {
  return RigTForm(tform.getTranslation());
}

inline RigTForm linFact(const RigTForm& tform) {
  return RigTForm(tform.getRotation());
}

inline Matrix4 rigTFormToMatrix(const RigTForm& tform) {
  Matrix4 m = Matrix4::makeTranslation(tform.getTranslation()) * quatToMatrix(tform.getRotation());
  return m;
}

#endif
