$('#art-sub').click(function () {
    $.ajax({
        url: "/abstract.html",  //提交给哪个url
        type: "POST",                 //请求方式
        data: $('#art-cont').serialize(),    //请求数据可以以字典的形式，此处是获取这个form表单中的所有
        traditional: true,             // 提交数据中有数组
        dataType: "JSON",              // 写了这个不用反序列化data，data就直接是对象
        success:function (data) {
            if(data.status){
                $("#abs-cont").val(data.abstract)
                //location.reload();     //刷新页面
            }else {
                $('#abs-cont').text(data.error);
            }
        }
    })
});
