<?php
/**
 * Description of alert
 *
 * @author durso
 */
namespace library\dom\structures;
use library\dom\structures\components;
use library\dom\elements\components\link;
use library\dom\elements\components\inline;
use library\dom\elements\components\block;



class alert extends components{
    public function __construct($className = "alert-danger") {
        $this->root = new block("div");
        $this->root->addClass("alert $className errorMsg");
    }
    public function create($error){
        $a = new link("&times;");
        $a->addClass("close");
        $a->attr("data-dismiss","alert");
        $this->root->addComponent($a);
        $this->components["a"] = $a;
        $span = new inline("span",$error);
        $this->root->addComponent($span);
        $this->components["span"] = $span;
    }
    public function save(){
        return $this->root;
    }
    
    
}