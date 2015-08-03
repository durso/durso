<?php
/**
 * Description of inline
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\components\intext;

class option extends intext{

    
    public function __construct($text=false,$value = false) {
        parent::__construct("option",$text);
        if($value){
            $this->setValue($value);
        }
    }
    public function setValue($value){
        $this->attr("value",$value);
    }

}
