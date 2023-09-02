#ifndef QMLMACROS_H
#define QMLMACROS_H

/*!
 *  \def QML_WRITABLE_PROPERTY(type, name)
 *  \ingroup QT_QML_HELPERS
 *  \hideinitializer
 *  \details Creates a \c Q_PROPERTY that will be readable / writable from QML.
 *
 *  \param type The C++ type of the property
 *  \param name The name for the property
 *  \param capitalName - capitalized name of the property to insert into function names
 *
 *  It generates for this goal :
 *  \code
 *      {type} m_{name}; // private member variable
 *      {type} get_{name} () const; // public getter method
 *      void set_{name} ({type}); // public setter slot
 *      void {name}Changed ({type}); // notifier signal
 *  \endcode
 *
 *  \b Note : Any change from either C++ or QML side will trigger the
 * notification.
 */
#define QML_WRITABLE_PROPERTY(type, name, capitalName)                                      \
protected:                                                                     \
    Q_PROPERTY(type name READ get ## capitalName WRITE set ## capitalName NOTIFY name ## Changed)  \
private:                                                                       \
    type _ ## name{ };                                                              \
  \
public:                                                                        \
    type get ## capitalName() const { return _ ## name; }                                  \
Q_SIGNALS:                                                                     \
    void name ## Changed(type name);                                               \
public Q_SLOTS:                                                                \
    void set ## capitalName(type name){                                                 \
        if (_ ## name != name) {                                                    \
            _ ## name = name;                                                         \
            emit name ## Changed(_ ## name);                                            \
        }                                                                          \
    }                                                                            \
  \
private:


/*!
 *  \def QML_READONLY_PROPERTY(type, name)
 *  \ingroup QT_QML_HELPERS
 *  \hideinitializer
 *  \details Creates a \c Q_PROPERTY that will be readable from QML and writable
 * from C++.
 *
 *  \param type The C++ type of the property
 *  \param name The name for the property
 *
 *  It generates for this goal :
 *  \code
 *      {type} m_{name}; // private member variable
 *      {type} get_{name} () const; // public getter method
 *      void update_{name} ({type}); // public setter method
 *      void {name}Changed ({type}); // notifier signal
 *  \endcode
 *
 *  \b Note : Any change from C++ side will trigger the notification to QML.
 */
#define QML_READONLY_PROPERTY(type, name, capitalName)                                      \
protected:                                                                     \
    Q_PROPERTY(type name READ get ##  capitalName NOTIFY name ## Changed)                   \
private:                                                                       \
    type _ ## name{ };                                                             \
  \
public:                                                                        \
    type get ##  capitalName() const { return _ ## name; }                                 \
    void set ## capitalName(const type& name){                                                 \
        if (_ ## name != name) {                                                    \
            _ ## name = name;                                                         \
            emit name ## Changed(_ ## name);                                            \
        }                                                                          \
    }                                                                   \
Q_SIGNALS:                                                                     \
    void name ## Changed(type name);                                               \
  \
private:


/*!
 *  \def QML_CONSTANT_PROPERTY(type, name)
 *  \ingroup QT_QML_HELPERS
 *  \hideinitializer
 *  \details Creates a \c Q_PROPERTY for a constant value exposed from C++ to
 * QML.
 *
 *  \param type The C++ type of the property
 *  \param name The name for the property
 *
 *  It generates for this goal :
 *  \code
 *      {type} m_{name}; // private member variable
 *      {type} get_{name} () const; // public getter method
 *  \endcode
 *
 *  \b Note : There is no change notifier because value is constant.
 */

#define QML_CONSTANT_PROPERTY(type, name, capitalizedName)                     \
protected:                                                                     \
    Q_PROPERTY(type name READ get ## capitalizedName CONSTANT)                     \
private:                                                                       \
    type m_ ## name;                                                               \
  \
public:                                                                        \
    type get ## capitalizedName() const { return m_ ## name; }                       \
  \
private:


#endif // QMLMACROS_H
